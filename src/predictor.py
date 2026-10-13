"""Model loading and prediction helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"
IMAGE_SIZE = (224, 224)

_model = None
_class_names: list[str] | None = None


class ModelNotReadyError(RuntimeError):
    """Raised when a trained model has not yet been created."""


def _load_artifacts():
    global _model, _class_names
    if _model is not None and _class_names is not None:
        return _model, _class_names

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise ModelNotReadyError(
            "The trained model is not available yet. Download the PlantVillage dataset, "
            "install the dependencies, and run `python train.py` as explained in README.md."
        )

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH)
        with CLASSES_PATH.open("r", encoding="utf-8") as file:
            _class_names = json.load(file)
        if not isinstance(_class_names, list) or not _class_names:
            raise ValueError("classes.json must contain a non-empty JSON list.")
        return _model, _class_names
    except Exception as exc:
        _model = None
        _class_names = None
        raise RuntimeError(f"Could not load the trained model: {exc}") from exc


def format_label(raw_label: str) -> tuple[str, str, str]:
    """Convert dataset labels like Tomato___Early_blight into readable labels."""
    parts = raw_label.split("___", maxsplit=1)
    if len(parts) == 2:
        crop = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    else:
        crop = "Unknown crop"
        disease = raw_label.replace("_", " ").strip()
    return crop, disease, f"{crop} — {disease}"


def care_guidance(disease: str) -> tuple[str, list[str]]:
    """Provide general, non-prescriptive next steps for the predicted category."""
    if "healthy" in disease.lower():
        return "The model's closest match is a healthy leaf.", [
            "Continue regular plant monitoring.",
            "Water according to the crop's needs and avoid waterlogged soil.",
            "Recheck any new spots, wilting, or unusual growth as the plant develops.",
        ]

    return "The model found a visual pattern similar to a disease category.", [
        "Inspect several leaves and the whole plant; one photo may not show the full problem.",
        "Where practical, avoid handling healthy plants immediately after affected plants and clean tools.",
        "Improve airflow and avoid unnecessary leaf wetness where appropriate for the crop.",
        "Confirm the diagnosis with a local agricultural extension professional before applying treatment.",
    ]


def predict_pil_image(image: Image.Image) -> dict[str, Any]:
    """Predict a class from a PIL image and return a JSON-serializable result."""
    model, class_names = _load_artifacts()
    prepared = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32)
    batch = np.expand_dims(pixels, axis=0)  # The model includes its own rescaling layer.
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]

    if probabilities.ndim != 1 or len(probabilities) != len(class_names):
        raise RuntimeError("The model output does not match models/classes.json. Retrain the model.")

    top_indices = np.argsort(probabilities)[::-1][: min(3, len(class_names))]
    best_index = int(top_indices[0])
    crop, disease, readable_label = format_label(class_names[best_index])
    advice_intro, advice = care_guidance(disease)

    top_predictions = []
    for index in top_indices:
        index = int(index)
        _, _, label = format_label(class_names[index])
        top_predictions.append({"label": label, "confidence": round(float(probabilities[index]) * 100, 2)})

    return {
        "prediction": readable_label,
        "raw_label": class_names[best_index],
        "crop": crop,
        "disease": disease,
        "confidence": round(float(probabilities[best_index]) * 100, 2),
        "health_status": "Healthy-like" if "healthy" in disease.lower() else "Disease-like",
        "top_predictions": top_predictions,
        "advice_intro": advice_intro,
        "advice": advice,
        "disclaimer": (
            "Educational prediction only. The model may be wrong, especially for photos taken in "
            "different lighting or field conditions. Confirm important decisions with a qualified expert."
        ),
    }
