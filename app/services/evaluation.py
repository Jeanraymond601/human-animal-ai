from pathlib import Path
from collections import defaultdict

import mlflow
from PIL import Image
from ultralytics import YOLO

from app.services.detector import TARGET_CLASSES


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "yolo11n.pt"
DATASET_PATH = BASE_DIR / "data" / "validation" / "data.yaml"

DATABASE_PATH = BASE_DIR / "mlflow.db"
ARTIFACT_DIR = BASE_DIR / "mlartifacts"

RESULTS_DIR = BASE_DIR / "results" / "evaluation"

DEVICE = "cpu"
IMAGE_SIZE = 640
DEFAULT_IOU_THRESHOLD = 0.50

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

mlflow.set_experiment("Human-Animal-AI")


# ============================================================
# CORRESPONDANCE DES CLASSES
# ============================================================

CLASS_NAME_TO_ID = {
    name: class_id
    for class_id, name in TARGET_CLASSES.items()
}


# ============================================================
# CALCUL DE L'IOU
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calcule l'intersection sur union de deux boites XYXY.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection = (
        intersection_width * intersection_height
    )

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LECTURE DES ANNOTATIONS YOLO
# ============================================================

def load_ground_truth(label_path, image_width, image_height):
    """
    Lit un fichier YOLO :
    class_id x_center y_center width height

    Retourne des boites en coordonnées XYXY pixels.
    Un fichier vide est valide : il signifie « aucun objet ».
    """

    label_path = Path(label_path)

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Annotation introuvable : {label_path}"
        )

    ground_truth = []

    with label_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Annotation invalide ligne {line_number} : "
                    "5 valeurs sont attendues."
                )

            try:
                class_id = int(float(values[0]))

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

            except ValueError as error:
                raise ValueError(
                    f"Valeurs invalides ligne {line_number}."
                ) from error

            if not all(
                0.0 <= value <= 1.0
                for value in (
                    x_center,
                    y_center,
                    width,
                    height,
                )
            ):
                raise ValueError(
                    f"Coordonnées hors limites ligne {line_number}."
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Dimensions invalides ligne {line_number}."
                )

            # Ignorer les classes que notre application
            # ne cherche pas à détecter.
            if class_id not in TARGET_CLASSES:
                continue

            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height

            ground_truth.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2],
            })

    return ground_truth


# ============================================================
# EVALUATION INDIVIDUELLE
# ============================================================

def evaluate_single_image(
    image_path,
    label_path,
    detections,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
    confidence=None,
):
    """
    Évalue les prédictions d'une seule image par comparaison
    avec les annotations réelles au format YOLO.

    Les prédictions doivent respecter le format de detector.py
    (clés « classe », « confiance », « coordonnees » en pixels XYXY).

    `confidence` (optionnel) est le seuil utilisé pour la détection ;
    il est seulement enregistré dans MLflow.

    Retourne :
        precision, recall, f1, mean_iou, TP, FP, FN,
        métriques par classe et Run ID MLflow.
    """

    image_path = Path(image_path)
    label_path = Path(label_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Annotation introuvable : {label_path}"
        )

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(
            "Le seuil IoU doit être compris entre 0 et 1."
        )

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    ground_truth = load_ground_truth(
        label_path,
        image_width,
        image_height,
    )

    predictions_by_class = defaultdict(list)
    truth_by_class = defaultdict(list)

    # --------------------------------------------------------
    # ORGANISATION DES ANNOTATIONS
    # --------------------------------------------------------

    for item in ground_truth:
        truth_by_class[item["class_id"]].append(
            item["box"]
        )

    # --------------------------------------------------------
    # ORGANISATION DES PREDICTIONS
    # --------------------------------------------------------

    for detection in detections:

        class_name = detection.get("classe")
        class_id = CLASS_NAME_TO_ID.get(class_name)

        if class_id is None:
            continue

        coordinates = detection.get("coordonnees", [])

        if len(coordinates) != 4:
            continue

        box = [
            float(value)
            for value in coordinates
        ]

        prediction_confidence = float(
            detection.get("confiance", 0.0)
        )

        predictions_by_class[class_id].append({
            "box": box,
            "confidence": prediction_confidence,
        })

    # --------------------------------------------------------
    # CORRESPONDANCE DES BOITES
    # --------------------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0

    matched_ious = []

    class_results = {}

    all_classes = sorted(
        set(truth_by_class.keys())
        | set(predictions_by_class.keys())
    )

    for class_id in all_classes:

        class_predictions = predictions_by_class[class_id]
        class_truth = truth_by_class[class_id]

        # Les prédictions les plus confiantes sont traitées
        # en premier.
        class_predictions = sorted(
            class_predictions,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        matched_truth = set()

        class_tp = 0
        class_fp = 0

        for prediction in class_predictions:

            best_iou = 0.0
            best_index = None

            for truth_index, truth_box in enumerate(class_truth):

                if truth_index in matched_truth:
                    continue

                current_iou = calculate_iou(
                    prediction["box"],
                    truth_box,
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = truth_index

            if (
                best_index is not None
                and best_iou >= iou_threshold
            ):
                class_tp += 1
                matched_truth.add(best_index)
                matched_ious.append(best_iou)

            else:
                class_fp += 1

        class_fn = len(class_truth) - len(matched_truth)

        class_precision = (
            class_tp / (class_tp + class_fp)
            if class_tp + class_fp > 0
            else 0.0
        )

        class_recall = (
            class_tp / (class_tp + class_fn)
            if class_tp + class_fn > 0
            else 0.0
        )

        class_f1 = (
            2 * class_precision * class_recall
            / (class_precision + class_recall)
            if class_precision + class_recall > 0
            else 0.0
        )

        class_name = TARGET_CLASSES[class_id]

        class_results[class_name] = {
            "TP": class_tp,
            "FP": class_fp,
            "FN": class_fn,
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
        }

        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn

    # --------------------------------------------------------
    # METRIQUES GLOBALES
    # --------------------------------------------------------

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_iou = (
        sum(matched_ious) / len(matched_ious)
        if matched_ious
        else 0.0
    )

    # Les noms de clés sont ceux lus par l'interface Streamlit.
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
    }

    # --------------------------------------------------------
    # ENREGISTREMENT MLFLOW
    # --------------------------------------------------------

    run_params = {
        "model": "YOLO11n",
        "image": image_path.name,
        "image_name": image_path.name,
        "annotation": label_path.name,
        "iou_threshold": iou_threshold,
        "device": DEVICE,
    }

    if confidence is not None:
        run_params["confidence_threshold"] = confidence

    with mlflow.start_run(
        run_name=f"Image-Evaluation-{image_path.stem}"
    ) as run:

        mlflow.set_tags({
            "evaluation_type": "single_image",
            "model": "YOLO11n",
            "image": image_path.name,
        })

        mlflow.log_params(run_params)

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            class_results,
            "evaluation/class_metrics.json",
        )

        mlflow.log_artifact(
            str(image_path),
            artifact_path="evaluation/images",
        )

        mlflow.log_artifact(
            str(label_path),
            artifact_path="evaluation/labels",
        )

        run_id = run.info.run_id

    return {
        **metrics,
        "classes": class_results,
        "run_id": run_id,
        "image": str(image_path),
        "annotation": str(label_path),
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence,
    }


# ============================================================
# EVALUATION GLOBALE DU DATASET
# ============================================================

def evaluate_model(
    model_path: Path = MODEL_PATH,
    data_yaml: Path = DATASET_PATH,
    imgsz: int = IMAGE_SIZE,
    batch: int = 1,
    device: str = DEVICE,
    conf: float = 0.001,
    iou: float = 0.6,
) -> dict:
    """
    Évalue YOLO11n sur le dataset complet.

    Retourne Precision, Recall, mAP50, mAP50-95,
    le dossier des résultats et le Run ID MLflow.
    """

    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Configuration dataset introuvable : {data_yaml}"
        )

    model = YOLO(str(model_path))

    with mlflow.start_run(
        run_name="YOLO11n-Dataset-Evaluation"
    ) as run:

        mlflow.set_tags({
            "evaluation_type": "dataset",
            "model": "YOLO11n",
        })

        mlflow.log_params({
            "model": model_path.name,
            "dataset": str(data_yaml),
            "image_size": imgsz,
            "batch_size": batch,
            "device": device,
            "confidence_threshold": conf,
            "iou_threshold": iou,
        })

        print("\n================================")
        print(" HUMAN-ANIMAL AI - EVALUATION")
        print("================================")
        print(f"Modèle : {model_path.name}")
        print(f"Dataset : {data_yaml}")
        print("Évaluation en cours...\n")

        validation_results = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            iou=iou,
            plots=True,
            verbose=True,
            project=str(RESULTS_DIR),
            name="yolo11n_validation",
            exist_ok=True,
        )

        precision = float(validation_results.box.mp)
        recall = float(validation_results.box.mr)
        map50 = float(validation_results.box.map50)
        map50_95 = float(validation_results.box.map)

        evaluation_metrics = {
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50_95": map50_95,
        }

        mlflow.log_metrics(evaluation_metrics)

        save_dir = Path(validation_results.save_dir)

        if save_dir.exists():
            mlflow.log_artifacts(
                str(save_dir),
                artifact_path="evaluation",
            )

        print("\n================================")
        print(" RESULTATS DE L'EVALUATION")
        print("================================")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"mAP@50    : {map50:.4f}")
        print(f"mAP@50-95 : {map50_95:.4f}")
        print(f"\nResultats : {save_dir}")
        print(f"MLflow Run ID : {run.info.run_id}")
        print("Evaluation terminee.")

        return {
            **evaluation_metrics,
            "save_dir": str(save_dir),
            "run_id": run.info.run_id,
        }


# ============================================================
# EXECUTION DIRECTE
# ============================================================

if __name__ == "__main__":
    evaluate_model()