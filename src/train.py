from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_new_model(img_size: int) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    """Build MobileNetV2 classifier with offline-safe weight loading."""
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        print("Loaded ImageNet weights for MobileNetV2.")
    except Exception as exc:
        print(f"Could not load ImageNet weights ({exc}). Using random initialization.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights=None,
        )

    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model, base_model


def find_mobilenet_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    return None


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    train_dir = base_dir / "dataset" / "training"
    val_dir = base_dir / "dataset" / "validation"
    models_dir = base_dir / "models"
    outputs_dir = base_dir / "outputs"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    img_size = 224
    batch_size = 16
    warmup_epochs = 5
    finetune_epochs = 8

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
    )

    class_mapping = {str(v): k for k, v in train_generator.class_indices.items()}
    with (outputs_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    model_path = models_dir / "fake_currency_mobilenetv2.keras"

    if model_path.exists():
        print(f"Found existing model at {model_path}. Continuing training from it.")
        model = tf.keras.models.load_model(model_path)
        base_model = find_mobilenet_backbone(model)
    else:
        model, base_model = build_new_model(img_size)

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    class_counts = np.bincount(train_generator.classes)
    total = class_counts.sum()
    class_weights = {
        0: total / (2.0 * class_counts[0]) if class_counts[0] > 0 else 1.0,
        1: total / (2.0 * class_counts[1]) if class_counts[1] > 0 else 1.0,
    }

    if base_model is not None:
        base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\nWarmup training...")
    history_warmup = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=warmup_epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    history_finetune = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers[:-25]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        print("\nFine-tuning...")
        fine_hist = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=finetune_epochs,
            class_weight=class_weights,
            callbacks=[checkpoint, early_stop],
            verbose=1,
        )
        history_finetune = fine_hist.history

    history = {
        "warmup": history_warmup.history,
        "finetune": history_finetune,
    }
    with (outputs_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Best model saved to: {model_path}")
    print(f"Class mapping saved to: {outputs_dir / 'class_names.json'}")


if __name__ == "__main__":
    main()
