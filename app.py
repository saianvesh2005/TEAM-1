from __future__ import annotations

import json
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Fake Currency Detector",
    layout="centered",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(uploaded_file, img_size: int) -> np.ndarray:
    img = tf.keras.utils.load_img(
        BytesIO(uploaded_file.getvalue()),
        target_size=(img_size, img_size),
    )
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def extract_text_with_tesseract(uploaded_file) -> str:
    # Run OCR via local tesseract binary and return normalized lowercase text.
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        result = subprocess.run(
            ["tesseract", tmp.name, "stdout", "-l", "eng"],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        return ""
    return result.stdout.lower()


def text_rule_for_fake(ocr_text: str) -> tuple[bool, str]:
    normalized = re.sub(r"\s+", " ", ocr_text).strip()
    trigger_phrases = ["children", "full of fun"]
    for phrase in trigger_phrases:
        if phrase in normalized:
            return True, phrase
    return False, ""


@st.cache_resource
def load_model_cached(model_path: Path):
    return tf.keras.models.load_model(model_path)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "fake_currency_mobilenetv2.keras"
    class_map_path = base_dir / "outputs" / "class_names.json"
    threshold_path = base_dir / "outputs" / "best_threshold.json"
    evaluation_path = base_dir / "outputs" / "evaluation.json"

    st.title("Fake Currency Detector")
    st.caption("Upload a note image to classify it as fake or real.")

    if not model_path.exists() or not class_map_path.exists():
        st.error("Model files are missing. Retrain first.")
        return

    class_mapping = load_json(class_map_path)

    threshold_default = 0.5
    optimized_label_index = 1
    if threshold_path.exists():
        threshold_data = load_json(threshold_path)
        threshold_default = float(threshold_data.get("best_threshold", 0.5))
        optimized_label_index = int(threshold_data.get("optimized_label_index", 1))

    model_accuracy = None
    if evaluation_path.exists():
        model_accuracy = float(load_json(evaluation_path).get("accuracy", 0.0))

    threshold = threshold_default

    col1, col2 = st.columns(2)
    with col1:
        if model_accuracy is not None:
            st.metric("Model Accuracy", f"{model_accuracy * 100:.2f}%")
        else:
            st.metric("Model Accuracy", "N/A")
    with col2:
        st.metric("Threshold (Auto)", f"{threshold * 100:.0f}%")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    model = load_model_cached(model_path)
    img_array = preprocess_image(uploaded_file, img_size=224)

    prob_class_1 = float(model.predict(img_array, verbose=0)[0][0])
    prob_class_0 = 1.0 - prob_class_1

    optimized_score = prob_class_1 if optimized_label_index == 1 else prob_class_0
    pred_index = (
        optimized_label_index
        if optimized_score >= threshold
        else (1 - optimized_label_index)
    )
    confidence = prob_class_1 if pred_index == 1 else prob_class_0

    ocr_text = extract_text_with_tesseract(uploaded_file)
    text_rule_hit, text_phrase = text_rule_for_fake(ocr_text)
    if text_rule_hit:
        fake_index = 0 if class_mapping["0"].lower() == "fake" else 1
        pred_index = fake_index
        confidence = 1.0

    pred_label = class_mapping[str(pred_index)]

    st.markdown("### Prediction")
    st.success(f"Predicted Class: {pred_label.upper()}")
    if text_rule_hit:
        st.warning(
            f"Text rule applied: detected phrase '{text_phrase}'. Marked as FAKE."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    with c2:
        st.metric(f"{class_mapping['0'].title()} Probability", f"{prob_class_0 * 100:.2f}%")
    with c3:
        st.metric(f"{class_mapping['1'].title()} Probability", f"{prob_class_1 * 100:.2f}%")


if __name__ == "__main__":
    main()
