"""
Gesture Recognition Module
Recognizes gestures from hand landmarks using rule-based classification.
"""

import time
from typing import Optional, Tuple
from collections import Counter, defaultdict, deque

from .hand_detector import HandLandmarks
from .config import GestureType, DetectionThresholds
from .utils import Point, MotionTracker, Debouncer


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

        self._majority_ratio = 0.55
        self._min_consecutive_frames = 2
        self._unknown_release_frames = 4
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

        hand_scale = self._hand_scale_factor(hand)

        if motion_gesture != GestureType.UNKNOWN:
            raw_gesture = motion_gesture
            gesture_type = "motion"
            confidence = 0.85
        else:
            # Check for static gestures
            static_gesture, confidence = self._detect_static_gesture(hand, hand_scale)
            raw_gesture = static_gesture

        if confidence < self.thresholds.gesture_confidence_threshold:
            raw_gesture = GestureType.UNKNOWN

        # Stabilize output to avoid rapid random switching.
        stabilized_gesture = self._stabilize_gesture(hand_id, raw_gesture)
        
        # Store for motion tracking
        self.previous_hands[hand_id] = hand
        
        return stabilized_gesture, {
            "type": gesture_type,
            "confidence": confidence,
            "raw_gesture": raw_gesture.value,
            "stable_gesture": stabilized_gesture.value,
            "hand_scale": hand_scale,
        }
    
    def _detect_static_gesture(self, hand: HandLandmarks, hand_scale: float) -> Tuple[GestureType, float]:
        """Detect static gesture from hand configuration."""
        pinch_threshold = self._scaled_threshold(self.thresholds.pinch_threshold, hand_scale)
        finger_threshold = self._scaled_threshold(self.thresholds.finger_up_threshold, hand_scale)
        state = self._finger_state(hand, finger_threshold)
        fingers_up_count = sum(state.values())

        index_up = state["index"]
        middle_up = state["middle"]
        ring_up = state["ring"]
        pinky_up = state["pinky"]
        non_thumb_up = sum([index_up, middle_up, ring_up, pinky_up])

        # Exact user rule: four folded fingers and only index lifted.
        exact_index_only = (
            state["index"]
            and not state["thumb"]
            and not state["middle"]
            and not state["ring"]
            and not state["pinky"]
        )
        if exact_index_only:
            return GestureType.INDEX_FINGER_UP, 0.86

        # Detect thumb-direction gestures (thumb only, others folded).
        thumb_dir_gesture, thumb_dir_conf = self._detect_thumb_direction_gesture(hand, state)
        if thumb_dir_gesture != GestureType.UNKNOWN:
            return thumb_dir_gesture, thumb_dir_conf
        
        # Check fist (all fingers closed)
        fist_closed = 5 - fingers_up_count
        if fist_closed >= 4:
            return GestureType.FIST, min(0.95, 0.65 + 0.08 * fist_closed)

        # Check pinch (thumb + index close together), but only when the hand is
        # not in an open-palm shape. This prevents open palm from being
        # repeatedly misclassified as pinch.
        pinch_distance = hand.thumb_tip.distance_2d(hand.index_finger_tip)
        x_min, y_min, x_max, y_max = hand.get_bounding_box()
        hand_size = max(x_max - x_min, y_max - y_min, 1e-6)
        pinch_ratio = pinch_distance / hand_size

        pinch_ok = (
            pinch_distance < (pinch_threshold * 1.15)
            and pinch_ratio < 0.38
            and (non_thumb_up <= 2 or (non_thumb_up == 3 and not index_up))
        )
        if pinch_ok:
            ratio = pinch_distance / max(pinch_threshold, 1e-6)
            confidence = max(0.60, min(0.98, 1.0 - (ratio * 0.28)))
            return GestureType.PINCH, confidence

        # Check open palm (all non-thumb fingers extended and clear thumb-index separation)
        open_palm_ok = (
            non_thumb_up >= 4
            and fingers_up_count >= 4
            and pinch_distance > (pinch_threshold * 1.45)
            and pinch_ratio > 0.22
        )

        # User rule: side gestures are a variant of open palm orientation.
        # Only open-palm-like states can become side_vertical/side_horizontal.
        if open_palm_ok:
            side_gesture, side_conf = self._detect_side_alignment_gesture(hand)
            if side_gesture != GestureType.UNKNOWN:
                return side_gesture, side_conf

        if open_palm_ok:
            return GestureType.OPEN_PALM, min(0.95, 0.62 + 0.07 * non_thumb_up)
        
        # Check index finger up (fallback check)
        if self._is_index_finger_up(hand, finger_threshold):
            return GestureType.INDEX_FINGER_UP, 0.78
        
        # Check two fingers (index and middle extended)
        if self._is_two_fingers(hand, finger_threshold):
            return GestureType.TWO_FINGERS, 0.74
        
        # Check three fingers (index, middle, ring extended)
        if self._is_three_fingers(hand, finger_threshold):
            return GestureType.THREE_FINGERS, 0.74

        return GestureType.UNKNOWN, 0.0

    def _detect_side_alignment_gesture(self, hand: HandLandmarks) -> Tuple[GestureType, float]:
        """Detect side-horizontal gesture from open-palm side alignment."""
        knuckle_span = hand.landmarks[5].distance_2d(hand.landmarks[17])
        palm_length = max(hand.landmarks[0].distance_2d(hand.landmarks[9]), 1e-6)
        side_ratio = knuckle_span / palm_length

        x_min, y_min, x_max, y_max = hand.get_bounding_box()
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)
        width_height_ratio = width / height

        fingertip_ids = (8, 12, 16, 20)
        tip_x = [hand.landmarks[i].x for i in fingertip_ids]
        tip_y = [hand.landmarks[i].y for i in fingertip_ids]
        tip_x_span = max(tip_x) - min(tip_x)
        tip_y_span = max(tip_y) - min(tip_y)
        tip_aspect = min(tip_x_span, tip_y_span) / max(max(tip_x_span, tip_y_span), 1e-6)

        # Strong edge-on constraints to avoid open-palm confusion.
        if side_ratio > 0.58:
            return GestureType.UNKNOWN, 0.0
        if 0.80 <= width_height_ratio <= 1.30:
            return GestureType.UNKNOWN, 0.0
        if tip_aspect > 0.55:
            return GestureType.UNKNOWN, 0.0

        wrist = hand.wrist
        middle_mcp = hand.landmarks[9]
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Require a dominant axis for stable side pose.
        if not (abs_dy > abs_dx * 1.15 or abs_dx > abs_dy * 1.15):
            return GestureType.UNKNOWN, 0.0

        confidence = float(max(0.62, min(0.96, 1.0 - side_ratio)))
        return GestureType.SIDE_HORIZONTAL, confidence
        return GestureType.UNKNOWN, 0.0

    def _detect_thumb_direction_gesture(self, hand: HandLandmarks, state: dict) -> Tuple[GestureType, float]:
        """Classify thumb-only poses into up/down/left/right directions."""
        non_thumb_all_down = not state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]
        if not (state["thumb"] and non_thumb_all_down):
            return GestureType.UNKNOWN, 0.0

        thumb_tip = hand.landmarks[4]
        thumb_mcp = hand.landmarks[2]
        dx = thumb_tip.x - thumb_mcp.x
        dy = thumb_tip.y - thumb_mcp.y

        magnitude = (dx * dx + dy * dy) ** 0.5
        if magnitude < 0.045:
            return GestureType.UNKNOWN, 0.0

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        confidence = max(0.64, min(0.96, magnitude * 9.0))

        if abs_dy > abs_dx * 1.15:
            return (GestureType.THUMB_UP, confidence) if dy < 0 else (GestureType.THUMB_DOWN, confidence)

        if abs_dx > abs_dy * 1.15:
            return (GestureType.THUMB_RIGHT, confidence) if dx > 0 else (GestureType.THUMB_LEFT, confidence)

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
    
    def _hand_scale_factor(self, hand: HandLandmarks) -> float:
        x_min, y_min, x_max, y_max = hand.get_bounding_box()
        hand_size = max(x_max - x_min, y_max - y_min)
        if not self.thresholds.adaptive_thresholds:
            return 1.0
        ref = max(self.thresholds.reference_hand_size, 1e-6)
        scale = hand_size / ref
        return max(self.thresholds.min_hand_scale, min(scale, self.thresholds.max_hand_scale))

    def _scaled_threshold(self, base: float, hand_scale: float) -> float:
        if not self.thresholds.adaptive_thresholds:
            return base
        return base * hand_scale

    def _finger_extended_ratio(self, hand: HandLandmarks, mcp_idx: int, pip_idx: int, tip_idx: int, ratio: float = 1.28, min_tip_pip: float = 0.04) -> bool:
        """Orientation-invariant extension test using bone length ratios."""
        mcp = hand.landmarks[mcp_idx]
        pip = hand.landmarks[pip_idx]
        tip = hand.landmarks[tip_idx]

        tip_to_mcp = tip.distance_2d(mcp)
        pip_to_mcp = max(pip.distance_2d(mcp), 1e-6)
        tip_to_pip = tip.distance_2d(pip)

        return (tip_to_mcp / pip_to_mcp) > ratio and tip_to_pip > min_tip_pip

    def _thumb_extended(self, hand: HandLandmarks, finger_threshold: float) -> bool:
        # Thumb bends differently; check if tip is significantly away from thumb MCP
        # and separated from the index base.
        thumb_tip = hand.thumb_tip
        thumb_mcp = hand.landmarks[2]
        thumb_ip = hand.thumb_ip
        index_mcp = hand.landmarks[5]

        tip_mcp = thumb_tip.distance_2d(thumb_mcp)
        ip_mcp = max(thumb_ip.distance_2d(thumb_mcp), 1e-6)
        thumb_spread = thumb_tip.distance_2d(index_mcp)

        return (tip_mcp / ip_mcp) > 1.18 and thumb_spread > max(0.04, 1.6 * finger_threshold)

    def _finger_wrist_extension(self, hand: HandLandmarks, pip_idx: int, tip_idx: int, margin: float = 0.012) -> bool:
        """Fallback extension check: tip farther from wrist than PIP by a margin."""
        wrist = hand.landmarks[0]
        pip = hand.landmarks[pip_idx]
        tip = hand.landmarks[tip_idx]
        return tip.distance_2d(wrist) > (pip.distance_2d(wrist) + margin)

    def _index_extended(self, hand: HandLandmarks) -> bool:
        # Primary ratio check + fallback wrist-relative check for perspective cases.
        primary = self._finger_extended_ratio(hand, 5, 6, 8, ratio=1.14, min_tip_pip=0.022)
        fallback = self._finger_wrist_extension(hand, 6, 8, margin=0.009)
        return primary or fallback

    def _finger_state(self, hand: HandLandmarks, finger_threshold: float) -> dict:
        return {
            "thumb": self._thumb_extended(hand, finger_threshold),
            # Index gets a custom rule because camera perspective often
            # underestimates its extension compared to other fingers.
            "index": self._index_extended(hand),
            "middle": self._finger_extended_ratio(hand, 9, 10, 12, ratio=1.18, min_tip_pip=0.028),
            "ring": self._finger_extended_ratio(hand, 13, 14, 16, ratio=1.14, min_tip_pip=0.024),
            "pinky": self._finger_extended_ratio(hand, 17, 18, 20, ratio=1.18, min_tip_pip=0.028),
        }

    def _count_fingers_up(self, hand: HandLandmarks, finger_threshold: float) -> int:
        return sum(self._finger_state(hand, finger_threshold).values())

    def _count_fingers_closed(self, hand: HandLandmarks, finger_threshold: float) -> int:
        return 5 - self._count_fingers_up(hand, finger_threshold)
    
    def _is_fist(self, hand: HandLandmarks, finger_threshold: float) -> bool:
        """Check if hand is closed in a fist."""
        state = self._finger_state(hand, finger_threshold)
        fingers_closed = [not state[k] for k in ["thumb", "index", "middle", "ring", "pinky"]]
        
        return sum(fingers_closed) >= 4  # At least 4 fingers closed
    
    def _is_open_palm(self, hand: HandLandmarks, finger_threshold: float) -> bool:
        """Check if hand is open with all fingers extended."""
        state = self._finger_state(hand, finger_threshold)
        fingers_up = [state[k] for k in ["thumb", "index", "middle", "ring", "pinky"]]
        
        return sum(fingers_up) >= 4  # At least 4 fingers extended
    
    def _is_index_finger_up(self, hand: HandLandmarks, finger_threshold: float) -> bool:
        """Check if only index finger is extended."""
        state = self._finger_state(hand, finger_threshold)
        return (
            state["index"]
            and not state["middle"]
            and not state["ring"]
            and not state["pinky"]
        )
    
    def _is_two_fingers(self, hand: HandLandmarks, finger_threshold: float) -> bool:
        """Check if index and middle fingers are extended."""
        state = self._finger_state(hand, finger_threshold)
        return (
            state["index"]
            and state["middle"]
            and not state["ring"]
            and not state["pinky"]
        )
    
    def _is_three_fingers(self, hand: HandLandmarks, finger_threshold: float) -> bool:
        """Check if index, middle, and ring fingers are extended."""
        state = self._finger_state(hand, finger_threshold)
        return (
            state["index"]
            and state["middle"]
            and state["ring"]
            and not state["pinky"]
        )
    
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
