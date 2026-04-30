"""
Pluggable face recognition engine.
Supports multiple backends: DeepFace, InsightFace (ArcFace), and legacy LBPH.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Result of a face detection operation."""
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: list[tuple[int, int]] | None = None
    aligned_face: np.ndarray | None = None


@dataclass
class RecognitionResult:
    """Result of a face recognition operation."""
    user_id: str | None
    user_name: str | None
    confidence: float
    embedding: np.ndarray | None = None


class FaceDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """Detect faces in an image.

        Args:
            image: BGR image array

        Returns:
            List of detection results
        """
        ...

    @abstractmethod
    def detect_largest(self, image: np.ndarray) -> DetectionResult | None:
        """Detect the largest face in an image."""
        ...


class FaceRecognizer(ABC):
    """Abstract base class for face recognizers."""

    @abstractmethod
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding (feature vector) from an aligned face image.

        Args:
            face_image: Aligned face image (RGB)

        Returns:
            Embedding vector (numpy array)
        """
        ...

    @abstractmethod
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings and return similarity score.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1, higher means more similar)
        """
        ...


# ── LBPH Backend (Legacy) ──────────────────────────────────────


class LBPHDetector(FaceDetector):
    """Face detector using OpenCV Haar cascade classifier."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        results = []
        for (x, y, w, h) in faces:
            results.append(DetectionResult(
                bounding_box=(x, y, w, h),
                confidence=1.0,
            ))
        return results

    def detect_largest(self, image: np.ndarray) -> DetectionResult | None:
        detections = self.detect(image)
        if not detections:
            return None
        return max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])


class LBPHRecognizer(FaceRecognizer):
    """Face recognizer using OpenCV LBPH algorithm (legacy baseline)."""

    def __init__(self):
        try:
            self._model = cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=8, grid_x=8, grid_y=8
            )
        except AttributeError:
            raise ImportError(
                "OpenCV face module not available. "
                "Install opencv-contrib-python: pip install opencv-contrib-python"
            )
        self._trained = False
        self._labels: dict[int, str] = {}

    def train(self, images: list[np.ndarray], labels: list[str]) -> None:
        """Train the LBPH model with face images and labels."""
        if not images:
            return
        int_labels = []
        label_set = {}
        next_id = 0
        for label in labels:
            if label not in label_set:
                label_set[label] = next_id
                self._labels[next_id] = label
                next_id += 1
            int_labels.append(label_set[label])
        self._model.train(images, np.array(int_labels))
        self._trained = True

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """For LBPH, the 'embedding' is the histogram vector."""
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        gray = cv2.resize(gray, (200, 200))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare using histogram correlation."""
        if len(embedding1) != len(embedding2):
            return 0.0
        return float(cv2.compareHist(
            embedding1.astype(np.float32),
            embedding2.astype(np.float32),
            cv2.HISTCMP_CORREL,
        ))

    def predict(self, face_image: np.ndarray) -> tuple[str | None, float]:
        """Predict the identity of a face image."""
        if not self._trained:
            return None, 0.0
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        gray = cv2.resize(gray, (200, 200))
        label_id, confidence = self._model.predict(gray)
        similarity = max(0.0, 1.0 - confidence / 100.0)
        user_id = self._labels.get(int(label_id))
        return user_id, similarity


# ── DeepFace Backend ───────────────────────────────────────────


class DeepFaceDetector(FaceDetector):
    """Face detector using DeepFace's built-in detectors."""

    def __init__(self, detector_backend: str = "opencv"):
        self._detector_backend = detector_backend

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        try:
            from deepface import DeepFace
            detections = DeepFace.extract_faces(
                image,
                detector_backend=self._detector_backend,
                enforce_detection=False,
            )
            results = []
            for det in detections:
                region = det.get("facial_area", {})
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)
                if w > 0 and h > 0:
                    results.append(DetectionResult(
                        bounding_box=(x, y, w, h),
                        confidence=det.get("confidence", 0.0),
                    ))
            return results
        except Exception:
            return []

    def detect_largest(self, image: np.ndarray) -> DetectionResult | None:
        detections = self.detect(image)
        if not detections:
            return None
        return max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])


class DeepFaceRecognizer(FaceRecognizer):
    """Face recognizer using DeepFace models."""

    def __init__(self, model_name: str = "ArcFace", detector_backend: str = "opencv"):
        self._model_name = model_name
        self._detector_backend = detector_backend

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        from deepface import DeepFace
        if face_image.shape[2] == 3:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = face_image
        embeddings = DeepFace.represent(
            rgb,
            model_name=self._model_name,
            detector_backend=self._detector_backend,
            enforce_detection=False,
        )
        if embeddings and len(embeddings) > 0:
            return np.array(embeddings[0]["embedding"], dtype=np.float32)
        return np.array([], dtype=np.float32)

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        dot = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return float(dot / (norm + 1e-8))


# ── InsightFace Backend (ArcFace) ─────────────────────────────


class InsightFaceDetector(FaceDetector):
    """Face detector using InsightFace SCRFD model."""

    def __init__(self, model_path: str = "data/models/"):
        self._model_path = model_path
        self._app = None

    def _init_app(self) -> Any:
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
                self._app = FaceAnalysis(name="buffalo_l", root=self._model_path)
                self._app.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize InsightFace: {e}")
        return self._app

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        try:
            app = self._init_app()
            faces = app.get(image)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                results.append(DetectionResult(
                    bounding_box=(x, y, x2 - x, y2 - y),
                    confidence=float(face.det_score),
                    landmarks=(
                        face.kps.tolist() if hasattr(face, "kps") else None
                    ),
                    aligned_face=(
                        face.normed_embedding
                        if hasattr(face, "normed_embedding")
                        else None
                    ),
                ))
            return results
        except Exception:
            return []

    def detect_largest(self, image: np.ndarray) -> DetectionResult | None:
        detections = self.detect(image)
        if not detections:
            return None
        return max(detections, key=lambda d: d.bounding_box[2] * d.bounding_box[3])


class InsightFaceRecognizer(FaceRecognizer):
    """Face recognizer using InsightFace ArcFace model."""

    def __init__(self, model_path: str = "data/models/"):
        self._model_path = model_path
        self._app = None

    def _init_app(self) -> Any:
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
                self._app = FaceAnalysis(name="buffalo_l", root=self._model_path)
                self._app.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize InsightFace: {e}")
        return self._app

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        app = self._init_app()
        if face_image.shape[2] == 3:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = face_image
        faces = app.get(rgb)
        if faces and len(faces) > 0:
            return faces[0].normed_embedding.astype(np.float32)
        return np.array([], dtype=np.float32)

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between two ArcFace embeddings."""
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        dot = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return float(dot / (norm + 1e-8))


# ── Factory ────────────────────────────────────────────────────


def create_detector(backend: str, **kwargs: Any) -> FaceDetector:
    """Factory function to create a face detector."""
    if backend == "lbph":
        return LBPHDetector()
    elif backend == "deepface":
        return DeepFaceDetector(detector_backend=kwargs.get("detector_backend", "opencv"))
    elif backend == "insightface":
        return InsightFaceDetector(model_path=kwargs.get("model_path", "data/models/"))
    else:
        raise ValueError(f"Unknown detector backend: {backend}")


def create_recognizer(backend: str, **kwargs: Any) -> FaceRecognizer:
    """Factory function to create a face recognizer."""
    if backend == "lbph":
        return LBPHRecognizer()
    elif backend == "deepface":
        return DeepFaceRecognizer(
            model_name=kwargs.get("model_name", "ArcFace"),
            detector_backend=kwargs.get("detector_backend", "opencv"),
        )
    elif backend == "insightface":
        return InsightFaceRecognizer(model_path=kwargs.get("model_path", "data/models/"))
    else:
        raise ValueError(f"Unknown recognizer backend: {backend}")
