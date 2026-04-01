"""Hand detection module using OpenCV skin-segmentation fallback."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import os

import cv2
import numpy as np

from .config import DetectionThresholds
from .utils import Point


@dataclass
class HandLandmarks:
    """Container for hand landmark-like points in normalized coordinates."""

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
    """Approximate hand detector that emits 21 pseudo-landmarks for the recognizer."""

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

        # Broad HSV skin range; this is intentionally permissive and may need tuning.
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        # Runtime filters to reduce flicker/noise.
        self._min_area_ratio = 0.012
        self._max_area_ratio = 0.22
        self._min_solidity = 0.60
        self._max_aspect_ratio = 2.2
        self._min_extent = 0.30
        self._required_stable_frames = 3
        self._stable_frames = 0
        self._prev_landmarks: Optional[List[Point]] = None
        self._landmark_smoothing_alpha = 0.35
        self._face_overlap_threshold = 0.12
        self._frames_without_hand = 0

        self._face_cascade = None
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            self._face_cascade = cascade

        self._profile_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "detector_profile.json",
        )
        self.profile_loaded = self.load_profile()

    def detect(self, frame: np.ndarray) -> Tuple[List[HandLandmarks], np.ndarray]:
        hands: List[HandLandmarks] = []
        annotated = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        # Blur helps remove isolated sensor noise before thresholding.
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Combine with YCrCb skin mask for stronger rejection of background regions.
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Detect face and exclude overlapping regions from hand candidates.
        face_rects: List[Tuple[int, int, int, int]] = []
        if self._face_cascade is not None:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            raw_faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )
            for fx, fy, fw, fh in raw_faces:
                expanded = self._expand_rect(fx, fy, fw, fh, frame_w, frame_h, pad_x=0.35, pad_y=0.45)
                face_rects.append(expanded)
                ex, ey, ew, eh = expanded
                cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process largest contours first to prioritize likely hand regions.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        relaxed_mode = self._frames_without_hand >= 8

        min_area_ratio = self._min_area_ratio * (0.6 if relaxed_mode else 1.0)
        min_solidity = max(0.25, self._min_solidity - (0.18 if relaxed_mode else 0.0))
        min_extent = max(0.10, self._min_extent - (0.15 if relaxed_mode else 0.0))
        max_aspect_ratio = self._max_aspect_ratio + (1.0 if relaxed_mode else 0.0)
        face_overlap_threshold = self._face_overlap_threshold + (0.25 if relaxed_mode else 0.0)

        for contour in contours:
            if len(hands) >= self.thresholds.max_num_hands:
                break

            area = cv2.contourArea(contour)
            area_ratio = area / frame_area
            if area_ratio < min_area_ratio or area_ratio > self._max_area_ratio:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue

            solidity = area / hull_area
            if solidity < min_solidity:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = bw / max(float(bh), 1.0)
            if aspect_ratio < 0.25 or aspect_ratio > max_aspect_ratio:
                continue

            rect_area = float(max(bw * bh, 1))
            extent = area / rect_area
            if extent < min_extent:
                continue

            contour_rejected = False
            contour_box = (x, y, bw, bh)
            for face_box in face_rects:
                if self._intersection_ratio(contour_box, face_box) > face_overlap_threshold:
                    contour_rejected = True
                    break
            if contour_rejected:
                continue

            landmarks = self._generate_pseudo_landmarks(contour, frame_w, frame_h)
            if len(landmarks) != 21:
                continue

            # Temporal stability gate: require a contour to persist for a few frames.
            self._stable_frames += 1
            if self._stable_frames < self._required_stable_frames:
                continue

            if self._prev_landmarks is not None and len(self._prev_landmarks) == 21:
                landmarks = self._smooth_landmarks(landmarks, self._prev_landmarks)

            self._prev_landmarks = landmarks

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
                handedness="Right",
                confidence=float(np.clip(0.5 + 0.5 * solidity, 0.5, 0.98)),
            )
            hands.append(hand)

            # Draw detector diagnostics on accepted contour only.
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"area={area_ratio:.3f} solid={solidity:.2f}",
                (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

        if not hands:
            self._stable_frames = 0
            self._prev_landmarks = None
            self._frames_without_hand += 1
        else:
            self._frames_without_hand = 0

        return hands, annotated

    def calibrate_from_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return False

        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

        h_ch = hsv[:, :, 0].reshape(-1)
        s_ch = hsv[:, :, 1].reshape(-1)
        v_ch = hsv[:, :, 2].reshape(-1)
        cr_ch = ycrcb[:, :, 1].reshape(-1)
        cb_ch = ycrcb[:, :, 2].reshape(-1)

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

        self.lower_ycrcb = np.array([
            0,
            int(np.percentile(cr_ch, 8)),
            int(np.percentile(cb_ch, 8)),
        ], dtype=np.uint8)
        self.upper_ycrcb = np.array([
            255,
            int(np.percentile(cr_ch, 92)),
            int(np.percentile(cb_ch, 92)),
        ], dtype=np.uint8)

        # Start with permissive geometry after calibration, then refine if needed.
        self._min_area_ratio = 0.003
        self._max_area_ratio = 0.35
        self._min_solidity = 0.40
        self._max_aspect_ratio = 2.8
        self._min_extent = 0.15
        self._required_stable_frames = 1
        self._face_overlap_threshold = 0.22
        self._frames_without_hand = 0
        return True

    def save_profile(self) -> bool:
        payload = {
            "lower_skin": self.lower_skin.tolist(),
            "upper_skin": self.upper_skin.tolist(),
            "lower_ycrcb": self.lower_ycrcb.tolist(),
            "upper_ycrcb": self.upper_ycrcb.tolist(),
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
            self.profile_loaded = True
            return True
        except Exception:
            return False

    def load_profile(self) -> bool:
        if not os.path.exists(self._profile_path):
            return False
        try:
            with open(self._profile_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)

            self.lower_skin = np.array(payload.get("lower_skin", self.lower_skin.tolist()), dtype=np.uint8)
            self.upper_skin = np.array(payload.get("upper_skin", self.upper_skin.tolist()), dtype=np.uint8)
            self.lower_ycrcb = np.array(payload.get("lower_ycrcb", self.lower_ycrcb.tolist()), dtype=np.uint8)
            self.upper_ycrcb = np.array(payload.get("upper_ycrcb", self.upper_ycrcb.tolist()), dtype=np.uint8)

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

    def _expand_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_w: int,
        frame_h: int,
        pad_x: float,
        pad_y: float,
    ) -> Tuple[int, int, int, int]:
        nx = max(0, int(x - w * pad_x))
        ny = max(0, int(y - h * pad_y))
        ex = min(frame_w - 1, int(x + w * (1.0 + pad_x)))
        ey = min(frame_h - 1, int(y + h * (1.0 + pad_y)))
        return nx, ny, max(ex - nx, 1), max(ey - ny, 1)

    def _intersection_ratio(
        self,
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = float((x2 - x1) * (y2 - y1))
        area_a = float(max(aw * ah, 1))
        return inter / area_a

    def _smooth_landmarks(self, current: List[Point], previous: List[Point]) -> List[Point]:
        alpha = self._landmark_smoothing_alpha
        return [
            Point(
                x=(1.0 - alpha) * p.x + alpha * c.x,
                y=(1.0 - alpha) * p.y + alpha * c.y,
                z=(1.0 - alpha) * p.z + alpha * c.z,
            )
            for c, p in zip(current, previous)
        ]

    def _generate_pseudo_landmarks(self, contour: np.ndarray, frame_w: int, frame_h: int) -> List[Point]:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 0:
            return []

        sampled = np.linspace(0, len(approx) - 1, 21, dtype=int)
        landmarks: List[Point] = []
        for idx in sampled:
            x, y = approx[idx][0]
            nx = float(np.clip(x / frame_w, 0.0, 1.0))
            ny = float(np.clip(y / frame_h, 0.0, 1.0))
            landmarks.append(Point(nx, ny, 0.0))

        return landmarks

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
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (5, 9),
            (9, 13),
            (13, 17),
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
        pass
