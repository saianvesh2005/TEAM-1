from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path: Path, img_size: int) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fake or real currency note.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/fake_currency_mobilenetv2.keras",
        help="Path to trained model",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    image_path = Path(args.image_path)
    model_path = base_dir / args.model
    class_map_path = base_dir / "outputs" / "class_names.json"
    threshold_path = base_dir / "outputs" / "best_threshold.json"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class mapping not found at {class_map_path}")

    class_mapping = load_json(class_map_path)
    threshold = 0.5
    optimized_label_index = 1
    if threshold_path.exists():
        threshold_data = load_json(threshold_path)
        threshold = float(threshold_data.get("best_threshold", 0.5))
        optimized_label_index = int(threshold_data.get("optimized_label_index", 1))

    model = tf.keras.models.load_model(model_path)

    img_array = preprocess_image(image_path, img_size=224)
    prob_class_1 = float(model.predict(img_array, verbose=0)[0][0])
    prob_class_0 = 1.0 - prob_class_1

    optimized_score = prob_class_1 if optimized_label_index == 1 else prob_class_0
    pred_index = optimized_label_index if optimized_score >= threshold else (1 - optimized_label_index)
    pred_label = class_mapping[str(pred_index)]
    confidence = prob_class_1 if pred_index == 1 else prob_class_0

    print("\nPrediction Result")
    print(f"Image: {image_path}")
    print(f"Threshold Used: {threshold:.2f}")
    print(f"Predicted Class: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probability ({class_mapping['0']}): {prob_class_0:.4f}")
    print(f"Probability ({class_mapping['1']}): {prob_class_1:.4f}")


if __name__ == "__main__":
    main()
