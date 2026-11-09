
from pathlib import Path
from typing import Any, cast

import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "yolo11n.pt"

# Classes COCO : personnes et animaux
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


# ==========================================
# 2. CHARGEMENT DU MODELE YOLO
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modele YOLO introuvable : {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("====================================")
print(" HUMAN & ANIMAL AI")
print("====================================")
print(f"Modele : {MODEL_PATH.name}")
print(f"Classes COCO disponibles : {len(model.names)}")
print("Moteur de detection initialise.")


# ==========================================
# 3. FONCTION DE DETECTION
# ==========================================

def detect_objects(
    image: Image.Image,
    confidence: float = 0.20
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:

    """
    Détecte les personnes et les animaux dans une image.

    Paramètres :
        image : image PIL à analyser.
        confidence : seuil de confiance entre 0 et 1.

    Retourne :
        annotated_image : image avec les boîtes de détection.
        detections : liste des objets détectés.
        counts : statistiques des détections.
    """

    # ======================================
    # VALIDATION
    # ======================================

    if not isinstance(image, Image.Image):
        raise TypeError(
            "L'image doit etre au format PIL."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "La confiance doit etre comprise entre 0 et 1."
        )

    # Conversion RGB
    image = image.convert("RGB")

    # ======================================
    # PREDICTION YOLO
    # ======================================

    results = cast(
        list[Results],
        model.predict(
            source=image,
            conf=confidence,
            iou=0.45,
            device="cpu",
            max_det=300,
            verbose=False
        )
    )

    # Récupération du premier résultat
    result = next(iter(results))

    # ======================================
    # PREPARATION DE L'IMAGE
    # ======================================

    image_bgr = cv2.cvtColor(
        __import__("numpy").array(image),
        cv2.COLOR_RGB2BGR
    )

    annotated = image_bgr.copy()

    # ======================================
    # INITIALISATION DES STATISTIQUES
    # ======================================

    detections: list[dict[str, Any]] = []

    counts: dict[str, Any] = {
        "personnes": 0,
        "animaux": 0,
        "total": 0,
        "par_classe": {}
    }

    # Couleurs BGR
    PERSON_COLOR = (255, 140, 0)
    ANIMAL_COLOR = (0, 200, 100)

    # ======================================
    # ANALYSE DES DETECTIONS
    # ======================================

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            # Ignorer les classes non recherchées
            if class_id not in TARGET_CLASSES:
                continue

            label = TARGET_CLASSES[class_id]

            # Coordonnées
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Catégorie
            category = (
                "Personne"
                if class_id == 0
                else "Animal"
            )

            # Couleur
            color = (
                PERSON_COLOR
                if class_id == 0
                else ANIMAL_COLOR
            )

            # ==================================
            # DESSIN DE LA BOITE
            # ==================================

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Etiquette
            text = f"{label} {score * 100:.1f}%"

            # Fond de l'étiquette
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            label_y = max(y1, text_height + 10)

            cv2.rectangle(
                annotated,
                (x1, label_y - text_height - 10),
                (x1 + text_width + 8, label_y),
                color,
                -1
            )

            cv2.putText(
                annotated,
                text,
                (x1 + 4, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ==================================
            # ENREGISTREMENT DE LA DETECTION
            # ==================================

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

            # ==================================
            # MISE A JOUR DES STATISTIQUES
            # ==================================

            counts["total"] += 1

            if class_id == 0:
                counts["personnes"] += 1

            elif class_id in ANIMAL_IDS:
                counts["animaux"] += 1

            counts["par_classe"][label] = (
                counts["par_classe"].get(label, 0) + 1
            )

    # ======================================
    # CONVERSION DE L'IMAGE ANNOTEE
    # ======================================

    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    annotated_image = Image.fromarray(
        annotated_rgb
    )

    # ======================================
    # RETOUR DES RESULTATS
    # ======================================

    return annotated_image, detections, counts
