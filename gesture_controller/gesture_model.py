"""Optional lightweight model-assist for hybrid gesture recognition.

This module intentionally keeps inference cheap: cosine similarity to per-class
feature centroids. It can bootstrap from file and self-calibrate online from
high-confidence rule predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GestureType
from .hand_detector import HandLandmarks


class GestureModelAssist:
    """Centroid-based gesture classifier with online adaptation."""

    SUPPORTED_GESTURES = {
        GestureType.TWO_FINGERS,
        GestureType.THREE_FINGERS,
        GestureType.THUMB_UP,
        GestureType.THUMB_DOWN,
        GestureType.THUMB_LEFT,
        GestureType.THUMB_RIGHT,
    }

    def __init__(
        self,
        model_path: str,
        min_samples: int = 8,
        min_confidence: float = 0.64,
        autosave_interval: int = 120,
    ):
        self.model_path = Path(model_path)
        self.min_samples = max(3, int(min_samples))
        self.min_confidence = float(min_confidence)
        self.autosave_interval = max(25, int(autosave_interval))

        self._centroids: Dict[str, np.ndarray] = {}
        self._counts: Dict[str, int] = {}
        self._updates_since_save = 0

        self._load()

    def _load(self):
        if not self.model_path.exists():
            return

        try:
            with open(self.model_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)

            classes = payload.get("classes", [])
            for item in classes:
                label = str(item.get("label", "")).strip()
                centroid = item.get("centroid")
                count = int(item.get("count", 0))
                if not label or centroid is None:
                    continue
                vec = np.array(centroid, dtype=np.float32)
                if vec.ndim != 1 or vec.size == 0:
                    continue
                self._centroids[label] = vec
                self._counts[label] = max(1, count)
        except Exception:
            # Keep model assist optional: fail-open and continue with rules only.
            self._centroids = {}
            self._counts = {}

    def _save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        classes = []
        for label, centroid in self._centroids.items():
            classes.append(
                {
                    "label": label,
                    "count": int(self._counts.get(label, 0)),
                    "centroid": centroid.astype(float).tolist(),
                }
            )

        payload = {
            "version": 1,
            "feature": "normalized_xy_landmarks",
            "classes": classes,
        }

        with open(self.model_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def _features(self, hand: HandLandmarks) -> np.ndarray:
        # Build translation- and scale-normalized XY features from all landmarks.
        wrist = hand.landmarks[0]
        x_min, y_min, x_max, y_max = hand.get_bounding_box()
        scale = max(x_max - x_min, y_max - y_min, 1e-6)

        feats: List[float] = []
        for lm in hand.landmarks:
            feats.append((lm.x - wrist.x) / scale)
            feats.append((lm.y - wrist.y) / scale)

        vec = np.array(feats, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def _gesture_to_label(self, gesture: GestureType) -> Optional[str]:
        if gesture in self.SUPPORTED_GESTURES:
            return gesture.value
        return None

    def update_from_rule(self, hand: HandLandmarks, gesture: GestureType, confidence: float):
        label = self._gesture_to_label(gesture)
        if label is None:
            return

        if confidence < 0.76:
            return

        vec = self._features(hand)

        if label not in self._centroids:
            self._centroids[label] = vec
            self._counts[label] = 1
        else:
            n = self._counts[label]
            centroid = self._centroids[label]
            updated = ((centroid * n) + vec) / float(n + 1)
            norm = float(np.linalg.norm(updated))
            if norm > 1e-6:
                updated = updated / norm
            self._centroids[label] = updated
            self._counts[label] = n + 1

        self._updates_since_save += 1
        if self._updates_since_save >= self.autosave_interval:
            self._updates_since_save = 0
            try:
                self._save()
            except Exception:
                pass

    def predict(self, hand: HandLandmarks) -> Tuple[GestureType, float]:
        if not self._centroids:
            return GestureType.UNKNOWN, 0.0

        vec = self._features(hand)

        scored: List[Tuple[str, float]] = []
        for label, centroid in self._centroids.items():
            if self._counts.get(label, 0) < self.min_samples:
                continue
            score = float(np.dot(vec, centroid))
            scored.append((label, score))

        if not scored:
            return GestureType.UNKNOWN, 0.0

        scored.sort(key=lambda it: it[1], reverse=True)
        best_label, best_score = scored[0]

        # Margin to reduce confusion among close classes.
        second = scored[1][1] if len(scored) > 1 else -1.0
        margin = best_score - second
        calibrated = max(0.0, min(1.0, (best_score + 1.0) * 0.5))
        calibrated *= max(0.0, min(1.0, margin * 2.2 + 0.35))

        if calibrated < self.min_confidence:
            return GestureType.UNKNOWN, calibrated

        try:
            return GestureType(best_label), calibrated
        except ValueError:
            return GestureType.UNKNOWN, 0.0

    def get_class_stats(self) -> Dict[str, Dict[str, int | bool]]:
        """Return per-class sample counts and readiness state."""
        stats: Dict[str, Dict[str, int | bool]] = {}

        for gesture in sorted(self.SUPPORTED_GESTURES, key=lambda g: g.value):
            label = gesture.value
            count = int(self._counts.get(label, 0))
            stats[label] = {
                "count": count,
                "ready": count >= self.min_samples,
            }

        return stats
