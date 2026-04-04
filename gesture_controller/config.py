"""
Configuration file for the Gesture Control System.
Defines gesture language, thresholds, and interaction modes.
"""

import enum
from dataclasses import dataclass
from typing import Dict, Tuple

# ===========================
# Gesture Definitions
# ===========================


class GestureType(enum.Enum):
    """Enumeration of all supported gesture types."""
    # Navigation gestures
    INDEX_FINGER_UP = "index_finger_up"
    PINCH = "pinch"
    
    # Control gestures
    OPEN_PALM = "open_palm"
    FIST = "fist"
    TWO_FINGERS = "two_fingers"
    THREE_FINGERS = "three_fingers"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    THUMB_LEFT = "thumb_left"
    THUMB_RIGHT = "thumb_right"

    # Side-alignment orientation gestures (detection-first)
    SIDE_HORIZONTAL = "side_horizontal"
    SIDE_VERTICAL = "side_vertical"
    
    # Motion-based gestures
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    
    # Continuous gestures
    PINCH_AND_MOVE = "pinch_and_move"
    PALM_MOVE = "palm_move"
    
    # Unknown gesture
    UNKNOWN = "unknown"


class InteractionMode(enum.Enum):
    """Enumeration of interaction modes."""
    IDLE = "idle"
    CONTROL = "control"
    SCROLL = "scroll"
    ZOOM = "zoom"
    LOCKED = "locked"


# ===========================
# Detection Thresholds
# ===========================


@dataclass
class DetectionThresholds:
    """Thresholds for hand and gesture detection."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Pinch detection: distance between thumb and index finger tip
    pinch_threshold: float = 0.038
    
    # Swipe detection parameters
    swipe_distance_threshold: float = 0.1  # Minimum distance to register a swipe
    swipe_time_threshold: float = 0.5  # Time window for swipe (seconds)
    
    # Finger state detection: threshold for finger "up" position
    finger_up_threshold: float = 0.025  # Relative to hand bounding box

    # Gesture confidence gate
    gesture_confidence_threshold: float = 0.58

    # Adaptive scaling (scale-invariant thresholds)
    adaptive_thresholds: bool = True
    reference_hand_size: float = 0.25
    min_hand_scale: float = 0.6
    max_hand_scale: float = 1.8
    
    # Motion smoothing
    smoothing_alpha: float = 0.45  # Exponential moving average factor
    
    # Debouncing
    debounce_delay: float = 0.22  # Seconds before same gesture can repeat

    # Optional model-assisted classification
    model_assist_enabled: bool = True
    model_assist_min_confidence: float = 0.64
    model_assist_min_samples: int = 8
    model_assist_autosave_interval: int = 120
    model_assist_path: str = "models/gesture_centroids.json"
    
    # Frame processing
    max_num_hands: int = 1
    static_image_mode: bool = False


@dataclass
class ScreenMapping:
    """Screen-related parameters for interaction."""
    screen_width: int = 2560  # Default screen width (pixels)
    screen_height: int = 1440  # Default screen height (pixels)
    
    # Coordinate mapping: hand position to cursor position
    # Typically, the entire video frame is mapped to a portion of the screen
    # Adjust these values based on your setup
    scale_x: float = 1.0  # Scaling factor for X coordinates
    scale_y: float = 1.0  # Scaling factor for Y coordinates
    offset_x: int = 0  # X offset for cursor positioning
    offset_y: int = 0  # Y offset for cursor positioning


# ===========================
# Gesture Mappings
# ===========================


GESTURE_ACTIONS: Dict[GestureType, str] = {
    GestureType.INDEX_FINGER_UP: "cursor_move",
    GestureType.PINCH: "click",
    GestureType.OPEN_PALM: "reset",
    GestureType.FIST: "fist_motion",
    GestureType.TWO_FINGERS: "toggle_playpause",
    GestureType.THREE_FINGERS: "switch_app",
    GestureType.THUMB_UP: "volume_up",
    GestureType.THUMB_DOWN: "volume_down",
    GestureType.THUMB_LEFT: "previous_tab",
    GestureType.THUMB_RIGHT: "next_tab",
    GestureType.SIDE_HORIZONTAL: "none",
    GestureType.SIDE_VERTICAL: "none",
    GestureType.SWIPE_LEFT: "previous_tab",
    GestureType.SWIPE_RIGHT: "next_tab",
    GestureType.SWIPE_UP: "volume_up",
    GestureType.SWIPE_DOWN: "volume_down",
    GestureType.PINCH_AND_MOVE: "drag",
    GestureType.PALM_MOVE: "scroll",
    GestureType.UNKNOWN: "none",
}


INTERACTION_MODE_TRANSITIONS: Dict[Tuple[InteractionMode, GestureType], InteractionMode] = {
    (InteractionMode.IDLE, GestureType.OPEN_PALM): InteractionMode.CONTROL,
    (InteractionMode.IDLE, GestureType.FIST): InteractionMode.LOCKED,
    (InteractionMode.CONTROL, GestureType.FIST): InteractionMode.LOCKED,
    (InteractionMode.CONTROL, GestureType.OPEN_PALM): InteractionMode.IDLE,
    (InteractionMode.LOCKED, GestureType.OPEN_PALM): InteractionMode.IDLE,
    (InteractionMode.SCROLL, GestureType.OPEN_PALM): InteractionMode.IDLE,
}


# ===========================
# System Configuration
# ===========================


@dataclass
class SystemConfig:
    """Overall system configuration."""
    # Display settings
    show_fps: bool = True
    show_landmarks: bool = True
    show_hand_info: bool = True
    show_gesture_label: bool = True
    
    # Processing settings
    enable_smoothing: bool = True
    enable_debouncing: bool = True
    
    # Recording and debugging
    record_video: bool = False
    debug_mode: bool = False
    
    # Performance
    target_fps: int = 30
    flip_frame: bool = True  # Flip frame horizontally for mirror effect


# ===========================
# Default Configuration
# ===========================


DEFAULT_THRESHOLDS = DetectionThresholds()
DEFAULT_SCREEN_MAPPING = ScreenMapping()
DEFAULT_SYSTEM_CONFIG = SystemConfig()
