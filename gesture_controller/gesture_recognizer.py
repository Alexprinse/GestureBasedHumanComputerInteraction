"""
Gesture Recognition Module
Recognizes gestures from hand landmarks using rule-based classification.
"""

import time
from typing import Optional, Tuple
from enum import Enum
from collections import Counter, defaultdict, deque

from .hand_detector import HandLandmarks
from .config import GestureType, DetectionThresholds
from .utils import Point, is_finger_up, MotionTracker, Debouncer


class GestureRecognizer:
    """
    Recognizes hand gestures using rule-based classification.
    
    Supports various gesture types:
    - Static gestures: pinch, fist, open palm, finger configurations
    - Motion gestures: swipes in all directions
    - Continuous gestures: pinch and move, palm move
    """
    
    def __init__(self, thresholds: DetectionThresholds = None):
        """
        Initialize the gesture recognizer.
        
        Args:
            thresholds: Detection thresholds
        """
        self.thresholds = thresholds or DetectionThresholds()
        
        # Motion tracking for swipe detection
        self.motion_trackers = {}  # Maps hand_id to MotionTracker
        self.hand_ids = set()
        
        # Debouncer to prevent rapid repeated gestures
        self.debouncer = Debouncer(self.thresholds.debounce_delay)
        
        # Time tracking
        self.last_frame_time = None
        self.swipe_start_time = {}
        
        # Previous hand state for motion detection
        self.previous_hands = {}

        # Temporal stabilization to reduce flicker/noisy class switching.
        self.gesture_history = defaultdict(lambda: deque(maxlen=7))
        self.stable_gesture = {}
        self.unknown_streak = defaultdict(int)
        self.motion_cooldown_until = defaultdict(float)

        self._majority_ratio = 0.60
        self._min_consecutive_frames = 3
        self._unknown_release_frames = 6
        self._motion_cooldown_seconds = 0.8
    
    def recognize(self, hand: HandLandmarks, hand_id: int = 0, 
                 current_time: Optional[float] = None) -> Tuple[GestureType, dict]:
        """
        Recognize a gesture from hand landmarks.
        
        Args:
            hand: HandLandmarks object
            hand_id: Unique identifier for this hand (for motion tracking)
            current_time: Current timestamp
            
        Returns:
            Tuple of (gesture_type, gesture_details)
            gesture_details dict contains additional info like confidence, motion, etc.
        """
        if current_time is None:
            current_time = time.time()
        
        # Update motion tracker
        if hand_id not in self.motion_trackers:
            self.motion_trackers[hand_id] = MotionTracker()
        
        hand_center = hand.get_center()
        self.motion_trackers[hand_id].add_position(hand_center, current_time)
        
        # Check for motion-based gestures (swipes)
        motion_gesture = self._detect_motion_gesture(hand_id, current_time)
        gesture_type = "static"
        confidence = 0.0
        raw_gesture = GestureType.UNKNOWN

        if motion_gesture != GestureType.UNKNOWN:
            raw_gesture = motion_gesture
            gesture_type = "motion"
            confidence = 0.85
        else:
            # Check for static gestures
            static_gesture, confidence = self._detect_static_gesture(hand)
            raw_gesture = static_gesture

        # Stabilize output to avoid rapid random switching.
        stabilized_gesture = self._stabilize_gesture(hand_id, raw_gesture)
        
        # Store for motion tracking
        self.previous_hands[hand_id] = hand
        
        return stabilized_gesture, {
            "type": gesture_type,
            "confidence": confidence,
            "raw_gesture": raw_gesture.value,
            "stable_gesture": stabilized_gesture.value,
        }
    
    def _detect_static_gesture(self, hand: HandLandmarks) -> Tuple[GestureType, float]:
        """Detect static gesture from hand configuration."""
        
        # Check pinch (thumb + index close together)
        if self._is_pinch(hand):
            return GestureType.PINCH, 0.9
        
        # Check fist (all fingers closed)
        if self._is_fist(hand):
            return GestureType.FIST, 0.85
        
        # Check open palm (all fingers extended)
        if self._is_open_palm(hand):
            return GestureType.OPEN_PALM, 0.85
        
        # Check index finger up (only index extended)
        if self._is_index_finger_up(hand):
            return GestureType.INDEX_FINGER_UP, 0.8
        
        # Check two fingers (index and middle extended)
        if self._is_two_fingers(hand):
            return GestureType.TWO_FINGERS, 0.75
        
        # Check three fingers (index, middle, ring extended)
        if self._is_three_fingers(hand):
            return GestureType.THREE_FINGERS, 0.75
        
        return GestureType.UNKNOWN, 0.0
    
    def _detect_motion_gesture(self, hand_id: int, current_time: float) -> GestureType:
        """Detect motion-based gestures (swipes)."""

        if current_time < self.motion_cooldown_until[hand_id]:
            return GestureType.UNKNOWN
        
        tracker = self.motion_trackers[hand_id]
        displacement = tracker.get_displacement()
        
        if displacement is None:
            return GestureType.UNKNOWN
        
        dx, dy = displacement
        distance = (dx**2 + dy**2)**0.5

        velocity = tracker.get_velocity()
        if velocity is None:
            return GestureType.UNKNOWN

        vx, vy = velocity
        speed = (vx**2 + vy**2)**0.5
        
        # Check if motion is significant enough
        if distance < (self.thresholds.swipe_distance_threshold * 1.35):
            return GestureType.UNKNOWN

        if speed < 0.45:
            return GestureType.UNKNOWN
        
        # Check if motion occurred within time window
        if hand_id not in self.swipe_start_time:
            self.swipe_start_time[hand_id] = current_time
            return GestureType.UNKNOWN
        
        time_elapsed = current_time - self.swipe_start_time[hand_id]
        
        if time_elapsed > self.thresholds.swipe_time_threshold:
            # Reset timer
            self.swipe_start_time[hand_id] = current_time
            return GestureType.UNKNOWN
        
        # Determine swipe direction based on displacement
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dx > abs_dy:
            # Horizontal swipe
            if abs_dy > 0 and (abs_dx / abs_dy) < 1.35:
                return GestureType.UNKNOWN
            if dx > 0:
                gesture = GestureType.SWIPE_RIGHT
            else:
                gesture = GestureType.SWIPE_LEFT
        else:
            # Vertical swipe
            if abs_dx > 0 and (abs_dy / abs_dx) < 1.35:
                return GestureType.UNKNOWN
            if dy > 0:
                gesture = GestureType.SWIPE_DOWN
            else:
                gesture = GestureType.SWIPE_UP
        
        # Reset tracker after swipe detection
        tracker.clear()
        self.swipe_start_time[hand_id] = current_time
        self.motion_cooldown_until[hand_id] = current_time + self._motion_cooldown_seconds
        
        return gesture

    def _stabilize_gesture(self, hand_id: int, raw_gesture: GestureType) -> GestureType:
        """Apply temporal majority voting + hold logic to stabilize labels."""
        history = self.gesture_history[hand_id]
        history.append(raw_gesture)

        current_stable = self.stable_gesture.get(hand_id, GestureType.UNKNOWN)

        if raw_gesture == GestureType.UNKNOWN:
            self.unknown_streak[hand_id] += 1
            if self.unknown_streak[hand_id] >= self._unknown_release_frames:
                self.stable_gesture[hand_id] = GestureType.UNKNOWN
                return GestureType.UNKNOWN
            return current_stable

        self.unknown_streak[hand_id] = 0

        non_unknown = [g for g in history if g != GestureType.UNKNOWN]
        if not non_unknown:
            return current_stable

        counts = Counter(non_unknown)
        candidate, candidate_count = counts.most_common(1)[0]
        majority_ok = (candidate_count / len(non_unknown)) >= self._majority_ratio
        consecutive_ok = self._tail_consecutive_count(non_unknown, candidate) >= self._min_consecutive_frames

        if majority_ok and consecutive_ok:
            self.stable_gesture[hand_id] = candidate
            return candidate

        return current_stable

    def _tail_consecutive_count(self, gestures, target: GestureType) -> int:
        count = 0
        for gesture in reversed(gestures):
            if gesture == target:
                count += 1
            else:
                break
        return count
    
    def _is_pinch(self, hand: HandLandmarks) -> bool:
        """Check if hand is in pinch position (thumb + index close)."""
        distance = hand.thumb_tip.distance_2d(hand.index_finger_tip)
        return distance < self.thresholds.pinch_threshold
    
    def _is_fist(self, hand: HandLandmarks) -> bool:
        """Check if hand is closed in a fist."""
        # All fingertips should be below their PIP joints (fingers closed)
        fingers_closed = [
            not is_finger_up(hand.thumb_tip, hand.thumb_pip, 
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.index_finger_tip, hand.landmarks[6],
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.middle_finger_tip, hand.landmarks[10],
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.ring_finger_tip, hand.landmarks[14],
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.pinky_tip, hand.landmarks[18],
                           self.thresholds.finger_up_threshold),
        ]
        
        return sum(fingers_closed) >= 4  # At least 4 fingers closed
    
    def _is_open_palm(self, hand: HandLandmarks) -> bool:
        """Check if hand is open with all fingers extended."""
        fingers_up = [
            is_finger_up(hand.thumb_tip, hand.thumb_pip,
                        self.thresholds.finger_up_threshold),
            is_finger_up(hand.index_finger_tip, hand.landmarks[6],
                        self.thresholds.finger_up_threshold),
            is_finger_up(hand.middle_finger_tip, hand.landmarks[10],
                        self.thresholds.finger_up_threshold),
            is_finger_up(hand.ring_finger_tip, hand.landmarks[14],
                        self.thresholds.finger_up_threshold),
            is_finger_up(hand.pinky_tip, hand.landmarks[18],
                        self.thresholds.finger_up_threshold),
        ]
        
        return sum(fingers_up) >= 4  # At least 4 fingers extended
    
    def _is_index_finger_up(self, hand: HandLandmarks) -> bool:
        """Check if only index finger is extended."""
        index_up = is_finger_up(hand.index_finger_tip, hand.landmarks[6],
                               self.thresholds.finger_up_threshold)
        
        other_fingers_down = [
            not is_finger_up(hand.middle_finger_tip, hand.landmarks[10],
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.ring_finger_tip, hand.landmarks[14],
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.pinky_tip, hand.landmarks[18],
                           self.thresholds.finger_up_threshold),
        ]
        
        return index_up and sum(other_fingers_down) >= 2
    
    def _is_two_fingers(self, hand: HandLandmarks) -> bool:
        """Check if index and middle fingers are extended."""
        index_up = is_finger_up(hand.index_finger_tip, hand.landmarks[6],
                               self.thresholds.finger_up_threshold)
        middle_up = is_finger_up(hand.middle_finger_tip, hand.landmarks[10],
                                self.thresholds.finger_up_threshold)
        
        other_fingers_down = [
            not is_finger_up(hand.ring_finger_tip, hand.landmarks[14],
                           self.thresholds.finger_up_threshold),
            not is_finger_up(hand.pinky_tip, hand.landmarks[18],
                           self.thresholds.finger_up_threshold),
        ]
        
        return index_up and middle_up and sum(other_fingers_down) >= 1
    
    def _is_three_fingers(self, hand: HandLandmarks) -> bool:
        """Check if index, middle, and ring fingers are extended."""
        index_up = is_finger_up(hand.index_finger_tip, hand.landmarks[6],
                               self.thresholds.finger_up_threshold)
        middle_up = is_finger_up(hand.middle_finger_tip, hand.landmarks[10],
                                self.thresholds.finger_up_threshold)
        ring_up = is_finger_up(hand.ring_finger_tip, hand.landmarks[14],
                              self.thresholds.finger_up_threshold)
        
        pinky_down = not is_finger_up(hand.pinky_tip, hand.landmarks[18],
                                      self.thresholds.finger_up_threshold)
        
        return index_up and middle_up and ring_up and pinky_down
    
    def reset(self):
        """Reset recognizer state."""
        self.motion_trackers.clear()
        self.swipe_start_time.clear()
        self.previous_hands.clear()
        self.gesture_history.clear()
        self.stable_gesture.clear()
        self.unknown_streak.clear()
        self.motion_cooldown_until.clear()
        self.debouncer.reset()
