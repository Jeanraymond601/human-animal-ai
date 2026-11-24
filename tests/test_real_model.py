"""Tests d'integration : detection avec le vrai modele YOLO11n."""

from pathlib import Path

import pytest
from PIL import Image

from app.services.detector import MODEL_PATH, detect_objects

BASE_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="yolo11n.pt introuvable"
)


def load(path):
    return Image.open(BASE_DIR / path).convert("RGB")


def test_person_and_elephant():
    """test.jpg : une femme a cote d'un elephant."""
    _, _, counts = detect_objects(load("test.jpg"), confidence=0.35)
    assert counts["personnes"] == 1
    assert counts["par_classe"].get("Elephant") == 1


def test_two_persons():
    """image2.jpg : deux personnes, aucun animal."""
    _, _, counts = detect_objects(
        load("data/validation/images/image2.jpg"), confidence=0.35
    )
    assert counts["personnes"] == 2
    assert counts["animaux"] == 0


def test_person_and_dog():
    """images.jpg : un homme et un chien (chien ~45 % de confiance)."""
    _, _, counts = detect_objects(
        load("data/validation/images/images.jpg"), confidence=0.35
    )
    assert counts["personnes"] == 1
    assert counts["par_classe"].get("Chien") == 1
