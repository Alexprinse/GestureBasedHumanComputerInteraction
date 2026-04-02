"""Hand detection module powered by MediaPipe Hand Landmarker Tasks API."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import json

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import DetectionThresholds
from .utils import Point


@dataclass
class HandLandmarks:
    """Container for hand landmarks in normalized coordinates."""

    landmarks: List[Point]
    wrist: Point
    index_finger_tip: Point
    middle_finger_tip: Point
    ring_finger_tip: Point
    pinky_tip: Point
    thumb_tip: Point
    thumb_ip: Point
    thumb_pip: Point
    handedness: str
    confidence: float

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        x_coords = [lm.x for lm in self.landmarks]
        y_coords = [lm.y for lm in self.landmarks]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def get_center(self) -> Point:
        x = sum(lm.x for lm in self.landmarks) / len(self.landmarks)
        y = sum(lm.y for lm in self.landmarks) / len(self.landmarks)
        z = sum(lm.z for lm in self.landmarks) / len(self.landmarks)
        return Point(x, y, z)


class HandDetector:
    """MediaPipe-based hand detector returning 21 anatomical landmarks."""

    WRIST = 0
    THUMB_PIP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_PIP = 14
    RING_TIP = 16
    PINKY_PIP = 18
    PINKY_TIP = 20

    def __init__(self, thresholds: DetectionThresholds = None):
        self.thresholds = thresholds or DetectionThresholds()

        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        self._profile_path = str(project_root / "detector_profile.json")

        model_path = models_dir / "hand_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. Run test setup to download it."
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.thresholds.max_num_hands,
            min_hand_detection_confidence=self.thresholds.min_detection_confidence,
            min_hand_presence_confidence=self.thresholds.min_tracking_confidence,
            min_tracking_confidence=self.thresholds.min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.profile_loaded = True

        # Compatibility fields for calibration UI.
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        self._min_area_ratio = 0.003
        self._max_area_ratio = 0.35
        self._min_solidity = 0.40
        self._max_aspect_ratio = 2.8
        self._min_extent = 0.15
        self._required_stable_frames = 1
        self._face_overlap_threshold = 0.22

    def detect(self, frame: np.ndarray) -> Tuple[List[HandLandmarks], np.ndarray]:
        hands: List[HandLandmarks] = []
        annotated = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return hands, annotated

        for idx, landmarks_list in enumerate(result.hand_landmarks):
            landmarks = [Point(lm.x, lm.y, lm.z) for lm in landmarks_list]
            if len(landmarks) != 21:
                continue

            handedness = "Unknown"
            confidence = 0.0
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                cat = result.handedness[idx][0]
                handedness = cat.category_name or "Unknown"
                confidence = float(cat.score)

            hand = HandLandmarks(
                landmarks=landmarks,
                wrist=landmarks[self.WRIST],
                thumb_tip=landmarks[self.THUMB_TIP],
                thumb_ip=landmarks[self.THUMB_IP],
                thumb_pip=landmarks[self.THUMB_PIP],
                index_finger_tip=landmarks[self.INDEX_TIP],
                middle_finger_tip=landmarks[self.MIDDLE_TIP],
                ring_finger_tip=landmarks[self.RING_TIP],
                pinky_tip=landmarks[self.PINKY_TIP],
                handedness=handedness,
                confidence=confidence,
            )
            hands.append(hand)

        return hands, annotated

    # Kept for compatibility with calibration UI. MediaPipe does not require this.
    def calibrate_from_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return False
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].reshape(-1)
        s_ch = hsv[:, :, 1].reshape(-1)
        v_ch = hsv[:, :, 2].reshape(-1)
        self.lower_skin = np.array([
            int(np.percentile(h_ch, 2)),
            int(np.percentile(s_ch, 12)),
            int(np.percentile(v_ch, 10)),
        ], dtype=np.uint8)
        self.upper_skin = np.array([
            int(np.percentile(h_ch, 98)),
            int(np.percentile(s_ch, 98)),
            int(np.percentile(v_ch, 98)),
        ], dtype=np.uint8)
        return True

    def save_profile(self) -> bool:
        payload = {
            "lower_skin": self.lower_skin.tolist(),
            "upper_skin": self.upper_skin.tolist(),
            "min_area_ratio": self._min_area_ratio,
            "max_area_ratio": self._max_area_ratio,
            "min_solidity": self._min_solidity,
            "max_aspect_ratio": self._max_aspect_ratio,
            "min_extent": self._min_extent,
            "required_stable_frames": self._required_stable_frames,
            "face_overlap_threshold": self._face_overlap_threshold,
        }
        try:
            with open(self._profile_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
            return True
        except Exception:
            return False

    def load_profile(self) -> bool:
        profile = Path(self._profile_path)
        if not profile.exists():
            return False
        try:
            with open(profile, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            self.lower_skin = np.array(payload.get("lower_skin", self.lower_skin.tolist()), dtype=np.uint8)
            self.upper_skin = np.array(payload.get("upper_skin", self.upper_skin.tolist()), dtype=np.uint8)
            self._min_area_ratio = float(payload.get("min_area_ratio", self._min_area_ratio))
            self._max_area_ratio = float(payload.get("max_area_ratio", self._max_area_ratio))
            self._min_solidity = float(payload.get("min_solidity", self._min_solidity))
            self._max_aspect_ratio = float(payload.get("max_aspect_ratio", self._max_aspect_ratio))
            self._min_extent = float(payload.get("min_extent", self._min_extent))
            self._required_stable_frames = int(payload.get("required_stable_frames", self._required_stable_frames))
            self._face_overlap_threshold = float(payload.get("face_overlap_threshold", self._face_overlap_threshold))
            return True
        except Exception:
            return False

    def get_profile_path(self) -> str:
        return self._profile_path

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hands: List[HandLandmarks],
        draw_connections: bool = True,
        draw_labels: bool = False,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        annotated = frame.copy()

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for hand in hands:
            for i, landmark in enumerate(hand.landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))

                if i == 0:
                    color, radius = (0, 255, 0), 6
                elif i in [4, 8, 12, 16, 20]:
                    color, radius = (0, 0, 255), 5
                else:
                    color, radius = (255, 0, 0), 3

                cv2.circle(annotated, (x, y), radius, color, -1)
                if draw_labels:
                    cv2.putText(
                        annotated,
                        str(i),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1,
                    )

            if draw_connections:
                for start_idx, end_idx in connections:
                    p1 = hand.landmarks[start_idx]
                    p2 = hand.landmarks[end_idx]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(annotated, (x1, y1), (x2, y2), (200, 100, 0), 2)

            x_min, y_min, _, _ = hand.get_bounding_box()
            cv2.putText(
                annotated,
                f"{hand.handedness} ({hand.confidence:.2f})",
                (int(x_min * w), max(int(y_min * h) - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated

    def get_hand_velocity(
        self,
        current_hand: HandLandmarks,
        previous_hand: Optional[HandLandmarks],
        dt: float,
    ) -> Optional[Tuple[float, float]]:
        if previous_hand is None or dt == 0:
            return None

        curr_center = current_hand.get_center()
        prev_center = previous_hand.get_center()
        vx = (curr_center.x - prev_center.x) / dt
        vy = (curr_center.y - prev_center.y) / dt
        return vx, vy

    def close(self):
        close_fn = getattr(self.detector, "close", None)
        if callable(close_fn):
            close_fn()
