
import time
from pathlib import Path
from PIL import Image

from app.services.detector import detect_objects
from app.services.tracking import track_detection


# ==========================================
# 1. CONFIGURATION DU TEST
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "test.jpg"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = RESULT_DIR / "test_result.jpg"

CONFIDENCE = 0.35


# ==========================================
# 2. VERIFICATION DE L'IMAGE
# ==========================================

print("\n================================")
print(" HUMAN & ANIMAL AI - TEST")
print("================================")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image de test introuvable : {IMAGE_PATH}"
    )

print("Image :", IMAGE_PATH.name)


# ==========================================
# 3. CHARGEMENT DE L'IMAGE
# ==========================================

image = Image.open(IMAGE_PATH).convert("RGB")


# ==========================================
# 4. DETECTION YOLO11n
# ==========================================

print("\nLancement de la detection...")

start_time = time.perf_counter()

annotated_image, detections, counts = detect_objects(
    image,
    confidence=CONFIDENCE
)

inference_time = time.perf_counter() - start_time

print("Detection terminee.")


# ==========================================
# 5. SAUVEGARDE DU RESULTAT
# ==========================================

annotated_image.save(RESULT_PATH)

print("Image annotee :", RESULT_PATH)


# ==========================================
# 6. AFFICHAGE DES STATISTIQUES
# ==========================================

print("\n========== RESULTATS ==========")

print("Personnes :", counts["personnes"])
print("Animaux :", counts["animaux"])
print("Total :", counts["total"])

print("\nDetail par classe :")

for label, number in counts["par_classe"].items():
    print(f"- {label} : {number}")

print("\nTemps d'inference :", round(inference_time, 3), "secondes")


# ==========================================
# 7. ENREGISTREMENT DANS MLFLOW
# ==========================================

print("\nEnregistrement dans MLflow...")

run_id = track_detection(
    image_name=IMAGE_PATH.name,
    confidence=CONFIDENCE,
    inference_time=inference_time,
    detections=detections,
    counts=counts,
    result_image=RESULT_PATH
)

print("Run ID :", run_id)


# ==========================================
# 8. FIN DU TEST
# ==========================================

print("\n================================")
print(" TEST TERMINE AVEC SUCCES")
print("================================")
