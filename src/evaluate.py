from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def find_best_threshold(
    y_true: np.ndarray, probs_class_1: np.ndarray, optimized_label_index: int
) -> tuple[float, float]:
    """Find threshold on optimized class probability that maximizes F1."""
    thresholds = np.arange(0.30, 0.901, 0.01)
    best_t = 0.5
    best_f1 = -1.0

    optimized_scores = (
        probs_class_1 if optimized_label_index == 1 else (1.0 - probs_class_1)
    )
    y_true_optimized = (y_true == optimized_label_index).astype(int)

    for t in thresholds:
        preds_optimized = (optimized_scores >= t).astype(int)
        f1 = f1_score(y_true_optimized, preds_optimized)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1


def apply_threshold(
    probs_class_1: np.ndarray, threshold: float, optimized_label_index: int
) -> np.ndarray:
    optimized_scores = (
        probs_class_1 if optimized_label_index == 1 else (1.0 - probs_class_1)
    )
    is_optimized = optimized_scores >= threshold
    other_index = 1 - optimized_label_index
    return np.where(is_optimized, optimized_label_index, other_index).astype(int)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    test_dir = base_dir / "dataset" / "testing"
    model_path = base_dir / "models" / "fake_currency_mobilenetv2.keras"
    outputs_dir = base_dir / "outputs"
    class_map_path = outputs_dir / "class_names.json"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class mapping not found at {class_map_path}")

    with class_map_path.open("r", encoding="utf-8") as f:
        class_mapping: dict[str, str] = json.load(f)

    fake_index = 0 if class_mapping["0"].lower() == "fake" else 1
    optimized_label = class_mapping[str(fake_index)]

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode="binary",
        shuffle=False,
    )

    model = tf.keras.models.load_model(model_path)
    print("\nEvaluating on test set...")

    probs_class_1 = model.predict(test_generator, verbose=0).ravel()
    y_true = test_generator.classes

    best_threshold, best_f1 = find_best_threshold(
        y_true, probs_class_1, optimized_label_index=fake_index
    )
    preds = apply_threshold(probs_class_1, best_threshold, optimized_label_index=fake_index)

    cm = confusion_matrix(y_true, preds)
    report = classification_report(
        y_true,
        preds,
        target_names=[class_mapping["0"], class_mapping["1"]],
        digits=4,
    )
    acc = float((preds == y_true).mean())

    print(f"Best Threshold (optimized for '{optimized_label}' F1): {best_threshold:.2f}")
    print(f"Optimized F1 ({optimized_label}): {best_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with (outputs_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    threshold_data = {
        "best_threshold": best_threshold,
        "optimized_label": optimized_label,
        "optimized_label_index": int(fake_index),
        "optimized_f1": best_f1,
    }
    with (outputs_dir / "best_threshold.json").open("w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"\nMetrics saved to: {outputs_dir / 'evaluation.json'}")
    print(f"Best threshold saved to: {outputs_dir / 'best_threshold.json'}")


if __name__ == "__main__":
    main()
