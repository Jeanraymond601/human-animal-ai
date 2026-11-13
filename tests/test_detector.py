
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from typing import Any, cast

from app.services import detector


class FakeBox:
    """Simule une boîte de détection YOLO."""

    def __init__(self, class_id, confidence, coordinates):
        self.cls = [np.array([class_id])]
        self.conf = [np.array([confidence])]
        self.xyxy = [np.array(coordinates, dtype=float)]


class FakeModel:
    """Simule le comportement du modèle YOLO."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)

        result = SimpleNamespace(
            boxes=self.boxes
        )

        return [result]


@pytest.fixture
def test_image():
    """Crée une image RGB de test."""
    return Image.new("RGB", (640, 480), color="white")


def install_fake_model(monkeypatch, boxes):
    """Remplace le modèle réel par un faux modèle."""
    fake_model = FakeModel(boxes)

    monkeypatch.setattr(
        detector,
        "get_model",
        lambda: fake_model
    )

    return fake_model



def test_invalid_image_type():
    """Une image qui n'est pas au format PIL doit être refusée."""

    invalid_image = cast(Any, "image.jpg")

    with pytest.raises(TypeError):
        detector.detect_objects(invalid_image)



@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1]
)
def test_invalid_confidence(test_image, confidence):
    """La confiance doit être comprise entre 0 et 1."""

    with pytest.raises(ValueError):
        detector.detect_objects(
            test_image,
            confidence=confidence
        )


def test_detect_person_and_animal(monkeypatch, test_image):
    """Vérifie la détection d'une personne et d'un animal."""

    boxes = [
        FakeBox(0, 0.95, [50, 60, 200, 300]),
        FakeBox(16, 0.88, [250, 100, 450, 350]),
        FakeBox(2, 0.90, [10, 10, 80, 80])
    ]

    fake_model = install_fake_model(monkeypatch, boxes)

    annotated, detections, counts = detector.detect_objects(
        test_image,
        confidence=0.20
    )

    # Vérifier l'image annotée.
    assert isinstance(annotated, Image.Image)
    assert annotated.mode == "RGB"
    assert annotated.size == test_image.size

    # Vérifier les détections.
    assert len(detections) == 2

    assert detections[0]["classe"] == "Personne"
    assert detections[1]["classe"] == "Chien"

    # Vérifier les statistiques.
    assert counts["personnes"] == 1
    assert counts["animaux"] == 1
    assert counts["total"] == 2

    assert counts["par_classe"] == {
        "Personne": 1,
        "Chien": 1
    }

    # Vérifier les paramètres d'inférence.
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["device"] == "cpu"
    assert fake_model.calls[0]["conf"] == 0.20


def test_no_detection(monkeypatch, test_image):
    """Vérifie le comportement lorsqu'aucun objet n'est détecté."""

    install_fake_model(monkeypatch, [])

    annotated, detections, counts = detector.detect_objects(
        test_image
    )

    assert isinstance(annotated, Image.Image)
    assert annotated.mode == "RGB"
    assert annotated.size == test_image.size

    assert detections == []

    assert counts == {
        "personnes": 0,
        "animaux": 0,
        "total": 0,
        "par_classe": {}
    }


def test_image_is_converted_to_rgb(monkeypatch):
    """Vérifie la conversion d'une image RGBA en RGB."""

    image = Image.new("RGBA", (320, 240), color="white")

    install_fake_model(monkeypatch, [])

    annotated, detections, counts = detector.detect_objects(image)

    assert annotated.mode == "RGB"
    assert annotated.size == (320, 240)
    assert detections == []
    assert counts["total"] == 0
