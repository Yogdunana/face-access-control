"""Tests for the Recognizer module."""
import numpy as np
import pytest

from src.core.recognizer import (
    DeepFaceRecognizer,
    LBPHDetector,
    LBPHRecognizer,
    create_detector,
    create_recognizer,
)

# Skip LBPH tests if opencv-contrib-python is not installed
try:
    import cv2
    cv2.face.LBPHFaceRecognizer_create()
    HAS_LBPH = True
except (AttributeError, Exception):
    HAS_LBPH = False


class TestLBPHDetector:
    def test_create(self):
        detector = LBPHDetector()
        assert detector is not None

    def test_detect_no_face(self):
        detector = LBPHDetector()
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(blank)
        assert len(results) == 0

    def test_detect_largest_no_face(self):
        detector = LBPHDetector()
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_largest(blank)
        assert result is None


@pytest.mark.skipif(not HAS_LBPH, reason="opencv-contrib-python not installed")
class TestLBPHRecognizer:
    def test_create(self):
        recognizer = LBPHRecognizer()
        assert recognizer is not None

    def test_extract_embedding(self):
        recognizer = LBPHRecognizer()
        face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        emb = recognizer.extract_embedding(face)
        assert emb is not None
        assert len(emb) > 0

    def test_compare_identical(self):
        recognizer = LBPHRecognizer()
        face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        emb = recognizer.extract_embedding(face)
        similarity = recognizer.compare(emb, emb)
        assert similarity > 0.9

    def test_compare_different(self):
        """LBPH: verify predict() correctly distinguishes different faces after training."""
        recognizer = LBPHRecognizer()
        # Create two visually distinct grayscale faces with different textures
        rng_a = np.random.RandomState(42)
        face_a = rng_a.randint(80, 130, (200, 200), dtype=np.uint8)
        rng_b = np.random.RandomState(99)
        face_b = rng_b.randint(150, 220, (200, 200), dtype=np.uint8)
        recognizer.train([face_a, face_b], ["person_a", "person_b"])

        # Predict face_a should return person_a
        pred_id, confidence = recognizer.predict(face_a)
        assert pred_id == "person_a"

        # Predict face_b should return person_b
        pred_id2, confidence2 = recognizer.predict(face_b)
        assert pred_id2 == "person_b"


class TestFactory:
    def test_create_lbph_detector(self):
        detector = create_detector("lbph")
        assert isinstance(detector, LBPHDetector)

    @pytest.mark.skipif(not HAS_LBPH, reason="opencv-contrib-python not installed")
    def test_create_lbph_recognizer(self):
        recognizer = create_recognizer("lbph")
        assert isinstance(recognizer, LBPHRecognizer)

    def test_create_deepface_recognizer(self):
        recognizer = create_recognizer("deepface", model_name="ArcFace")
        assert isinstance(recognizer, DeepFaceRecognizer)

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            create_detector("unknown")
        with pytest.raises(ValueError):
            create_recognizer("unknown")
