
import mlflow
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

# Base de données SQLite
DATABASE_PATH = BASE_DIR / "mlflow.db"

# Dossier de stockage des artefacts
ARTIFACT_DIR = BASE_DIR / "mlartifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

# Création ou récupération de l'expérience
mlflow.set_experiment("Human-Animal-AI")


# ==========================================
# 2. SUIVI DES EXPÉRIENCES
# ==========================================

def track_detection(
    image_name,
    confidence,
    inference_time,
    detections,
    counts,
    result_image
):
    """
    Enregistre une expérience de détection dans MLflow.
    """

    with mlflow.start_run(
        run_name=f"Detection-{image_name}"
    ) as run:

        # ==================================
        # 3. PARAMÈTRES DU MODÈLE
        # ==================================

        mlflow.log_param("model", "YOLO11n")
        mlflow.log_param("image", image_name)
        mlflow.log_param("confidence_threshold", confidence)
        mlflow.log_param("device", "CPU")

        # ==================================
        # 4. MÉTRIQUES DE PERFORMANCE
        # ==================================

        mlflow.log_metric(
            "inference_time_seconds",
            inference_time
        )

        mlflow.log_metric(
            "total_detections",
            counts["total"]
        )

        mlflow.log_metric(
            "persons_detected",
            counts["personnes"]
        )

        mlflow.log_metric(
            "animals_detected",
            counts["animaux"]
        )

        # Confiance moyenne
        if detections:
            average_confidence = sum(
                detection["confiance"]
                for detection in detections
            ) / len(detections)

            mlflow.log_metric(
                "average_confidence",
                average_confidence
            )

        # ==================================
        # 5. STATISTIQUES PAR CLASSE
        # ==================================

        for label, number in counts["par_classe"].items():

            metric_name = (
                "class_"
                + label.lower().replace(" ", "_")
            )

            mlflow.log_metric(metric_name, number)

        # ==================================
        # 6. SAUVEGARDE DES STATISTIQUES
        # ==================================

        mlflow.log_dict(
            counts,
            "results/counts.json"
        )

        # ==================================
        # 7. SAUVEGARDE DE L'IMAGE
        # ==================================

        image_path = Path(result_image)

        if image_path.exists():
            mlflow.log_artifact(
                str(image_path.resolve()),
                artifact_path="predictions"
            )
        else:
            print(
                f"Attention : image introuvable : {image_path}"
            )

        # ==================================
        # 8. CONFIRMATION DE L'EXPÉRIENCE
        # ==================================

        print("\n================================")
        print(" EXPÉRIENCE MLFLOW ENREGISTRÉE")
        print("================================")

        print("Expérience : Human-Animal-AI")
        print("Modèle : YOLO11n")
        print("Image :", image_name)
        print("Run ID :", run.info.run_id)

        print("================================\n")

        return run.info.run_id
