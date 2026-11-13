
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "yolo11n.pt"

TARGET_CLASSES = {
    0: "Personne",
    14: "Oiseau",
    15: "Chat",
    16: "Chien",
    17: "Cheval",
    18: "Mouton",
    19: "Vache",
    20: "Elephant",
    21: "Ours",
    22: "Zebre",
    23: "Girafe",
}

ANIMAL_IDS = set(TARGET_CLASSES.keys()) - {0}

# Le modèle sera chargé uniquement lors de la première détection.
_model = None


def get_model():
    """Charge YOLO une seule fois, au premier appel."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modele YOLO introuvable : {MODEL_PATH}"
            )

        _model = YOLO(str(MODEL_PATH))

        print("====================================")
        print(" HUMAN & ANIMAL AI")
        print("====================================")
        print(f"Modele : {MODEL_PATH.name}")
        print(f"Classes COCO disponibles : {len(_model.names)}")
        print("Moteur de detection initialise.")

    return _model


def detect_objects(
    image: Image.Image,
    confidence: float = 0.20
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:
    """
    Détecte les personnes et les animaux dans une image.

    Returns:
        annotated_image: image annotée avec les boîtes.
        detections: liste des objets détectés.
        counts: statistiques des détections.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("L'image doit etre au format PIL.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "La confiance doit etre comprise entre 0 et 1."
        )

    image = image.convert("RGB")

    results = cast(
        list[Results],
        get_model().predict(
            source=image,
            conf=confidence,
            iou=0.45,
            device="cpu",
            max_det=300,
            verbose=False
        )
    )

    if not results:
        raise RuntimeError("Le modele n'a retourne aucun resultat.")

    result = results[0]

    image_array = np.array(image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    annotated = image_bgr.copy()

    detections: list[dict[str, Any]] = []

    counts: dict[str, Any] = {
        "personnes": 0,
        "animaux": 0,
        "total": 0,
        "par_classe": {}
    }

    PERSON_COLOR = (255, 140, 0)
    ANIMAL_COLOR = (0, 200, 100)

    if result.boxes is not None:
        for box in result.boxes:

            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            # Ignorer les classes non ciblées.
            if class_id not in TARGET_CLASSES:
                continue

            label = TARGET_CLASSES[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            category = (
                "Personne" if class_id == 0 else "Animal"
            )

            color = (
                PERSON_COLOR if class_id == 0
                else ANIMAL_COLOR
            )

            # Dessiner la boîte de détection.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Préparer le texte.
            text = f"{label} {score * 100:.1f}%"

            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            label_y = max(y1, text_height + 10)

            # Fond du label.
            cv2.rectangle(
                annotated,
                (x1, label_y - text_height - 10),
                (x1 + text_width + 8, label_y),
                color,
                -1
            )

            # Texte du label.
            cv2.putText(
                annotated,
                text,
                (x1 + 4, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Enregistrer la détection.
            detection = {
                "classe": label,
                "categorie": category,
                "confiance": round(score * 100, 2),
                "coordonnees": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]
            }

            detections.append(detection)

            # Mettre à jour les statistiques.
            counts["total"] += 1

            if class_id == 0:
                counts["personnes"] += 1
            elif class_id in ANIMAL_IDS:
                counts["animaux"] += 1

            counts["par_classe"][label] = (
                counts["par_classe"].get(label, 0) + 1
            )

    # Conversion BGR vers RGB.
    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    annotated_image = Image.fromarray(annotated_rgb)

    return annotated_image, detections, counts
