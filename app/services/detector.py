from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "yolo11n.pt"

DEVICE = "cpu"
IMAGE_SIZE = 640
MODEL_IOU = 0.45
MAX_DETECTIONS = 300

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

PERSON_COLOR = (255, 140, 0)
ANIMAL_COLOR = (0, 200, 100)

TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 2

# Le modèle est chargé uniquement à la première détection.
_model: YOLO | None = None


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

def get_model() -> YOLO:
    """
    Charge YOLO11n une seule fois.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modele YOLO introuvable : {MODEL_PATH}"
            )

        _model = YOLO(str(MODEL_PATH))

        print("=" * 50)
        print(" HUMAN & ANIMAL AI")
        print("=" * 50)
        print(f"Modele : {MODEL_PATH.name}")
        print(f"Classes COCO disponibles : {len(_model.names)}")
        print(f"Appareil : {DEVICE}")
        print("Moteur de detection initialise.")

    return _model


# ============================================================
# VALIDATION DES RESULTATS
# ============================================================

def get_prediction_result(
    model: Any,
    image: Image.Image,
    confidence: float = 0.20,
) -> Any:
    """
    Execute l'inference et retourne un résultat valide.

    Accepte les résultats Ultralytics et les objets simulés
    utilisés dans les tests unitaires.
    """

    prediction_results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=confidence,
        iou=MODEL_IOU,
        device=DEVICE,
        max_det=MAX_DETECTIONS,
        verbose=False,
        stream=False,
    )

    result = next(iter(prediction_results), None)

    if result is None or not hasattr(result, "boxes"):
        raise RuntimeError(
            "Le modele n'a retourne aucun resultat valide."
        )

    return result


# ============================================================
# ANNOTATION DES DETECTIONS
# ============================================================

def draw_detection(
    image: np.ndarray,
    bbox: list[float],
    label: str,
    confidence: float,
    color: tuple[int, int, int],
) -> None:
    """
    Dessine une boite et son etiquette sur une image BGR.
    """

    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    # Limiter les coordonnées aux dimensions de l'image.
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        BOX_THICKNESS,
    )

    text = f"{label} {confidence * 100:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    padding = 4

    # Positionner le label au-dessus de la boîte si possible.
    label_top = y1 - text_height - baseline - padding * 2

    if label_top < 0:
        label_top = y1

    label_bottom = label_top + text_height + baseline + padding * 2

    label_right = min(
        width - 1,
        x1 + text_width + padding * 2,
    )

    label_bottom = min(height - 1, label_bottom)

    cv2.rectangle(
        image,
        (x1, label_top),
        (label_right, label_bottom),
        color,
        -1,
    )

    text_y = min(
        height - 1,
        label_top + text_height + padding,
    )

    cv2.putText(
        image,
        text,
        (x1 + padding, text_y),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# DETECTION PRINCIPALE
# ============================================================

def detect_objects(
    image: Image.Image,
    confidence: float = 0.20,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:
    """
    Détecte les personnes et les animaux dans une image.

    Args:
        image: image au format PIL.
        confidence: seuil minimal de confiance entre 0 et 1.

    Returns:
        annotated_image: image annotée avec les boîtes.
        detections: liste des objets détectés.
        counts: statistiques des détections.
    """

    # --------------------------------------------------------
    # VALIDATION DE L'IMAGE
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):
        raise TypeError(
            "L'image doit etre au format PIL."
        )

    if not isinstance(confidence, (int, float)):
        raise TypeError(
            "La confiance doit etre numerique."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "La confiance doit etre comprise entre 0 et 1."
        )

    image = image.convert("RGB")

    # --------------------------------------------------------
    # CHARGEMENT ET INFERENCE
    # --------------------------------------------------------

    model = get_model()

    result = get_prediction_result(
        model=model,
        image=image,
        confidence=float(confidence),
    )

    # --------------------------------------------------------
    # PREPARATION DE L'IMAGE
    # --------------------------------------------------------

    image_array = np.array(image)

    annotated = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR,
    )

    detections: list[dict[str, Any]] = []

    counts: dict[str, Any] = {
        "personnes": 0,
        "animaux": 0,
        "total": 0,
        "par_classe": {},
    }

    # --------------------------------------------------------
    # EXTRACTION DES BOITES
    # --------------------------------------------------------

    boxes = result.boxes

    if boxes is not None:

        # Compatibilité entre les objets FakeBox des tests
        # et les résultats réels de YOLO.

        detection_data: Iterable[tuple[Any, Any, Any]]

        if isinstance(boxes, list):

            detection_data = [
                (
                    box.cls[0],
                    box.conf[0],
                    box.xyxy[0],
                )
                for box in boxes
            ]

        else:

            class_ids = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            coordinates = boxes.xyxy.tolist()

            detection_data = zip(
                class_ids,
                confidences,
                coordinates,
            )

        for class_id, score, bbox in detection_data:

            # Conversion robuste des valeurs NumPy.
            class_id = int(np.asarray(class_id).reshape(-1)[0])
            score = float(np.asarray(score).reshape(-1)[0])
            bbox = np.asarray(bbox, dtype=float).reshape(-1).tolist()

            # Ignorer les classes non ciblées.
            if class_id not in TARGET_CLASSES:
                continue

            label = TARGET_CLASSES[class_id]

            bbox_values = [
                float(value)
                for value in bbox
            ]

            # Catégorie et couleur.
            if class_id == 0:
                category = "Personne"
                color = PERSON_COLOR
            else:
                category = "Animal"
                color = ANIMAL_COLOR

            # Annotation.
            draw_detection(
                image=annotated,
                bbox=bbox_values,
                label=label,
                confidence=score,
                color=color,
            )

            # Enregistrement.
            detection = {
                "classe": label,
                "categorie": category,
                "confiance": round(score * 100, 2),
                "coordonnees": [
                    round(value, 2)
                    for value in bbox_values
                ],
            }

            detections.append(detection)

            # Statistiques.
            counts["total"] += 1

            if class_id == 0:
                counts["personnes"] += 1

            elif class_id in ANIMAL_IDS:
                counts["animaux"] += 1

            counts["par_classe"][label] = (
                counts["par_classe"].get(label, 0) + 1
            )

    # --------------------------------------------------------
    # CONVERSION BGR VERS RGB
    # --------------------------------------------------------

    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB,
    )

    annotated_image = Image.fromarray(annotated_rgb)

    return annotated_image, detections, counts