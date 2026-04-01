"""
Utility functions for the Gesture Control System.
Includes mathematical operations, smoothing, and helper functions.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class Point:
    """Represents a 2D or 3D point."""
    
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def distance_to(self, other: "Point") -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def distance_2d(self, other: "Point") -> float:
        """Calculate 2D Euclidean distance (ignoring Z)."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self) -> str:
        return f"Point({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)


class ExponentialMovingAverage:
    """Exponential Moving Average for smoothing."""
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize the EMA filter.
        
        Args:
            alpha: Smoothing factor (0-1). Higher values give more weight to recent data.
        """
        self.alpha = alpha
        self.smoothed_value = None
    
    def update(self, value: float) -> float:
        """
        Update the filter with a new value and return smoothed result.
        
        Args:
            value: New data point
            
        Returns:
            Smoothed value
        """
        if self.smoothed_value is None:
            self.smoothed_value = value
        else:
            self.smoothed_value = self.alpha * value + (1 - self.alpha) * self.smoothed_value
        
        return self.smoothed_value
    
    def reset(self):
        """Reset the filter."""
        self.smoothed_value = None


class PointSmoother:
    """Smooth point trajectories using exponential moving average."""
    
    def __init__(self, alpha: float = 0.7):
        self.ema_x = ExponentialMovingAverage(alpha)
        self.ema_y = ExponentialMovingAverage(alpha)
        self.ema_z = ExponentialMovingAverage(alpha)
    
    def smooth(self, point: Point) -> Point:
        """Smooth a point using EMA."""
        return Point(
            self.ema_x.update(point.x),
            self.ema_y.update(point.y),
            self.ema_z.update(point.z)
        )
    
    def reset(self):
        """Reset all EMA filters."""
        self.ema_x.reset()
        self.ema_y.reset()
        self.ema_z.reset()


class Debouncer:
    """Debounce repeated gestures within a time window."""
    
    def __init__(self, debounce_delay: float = 0.2):
        """
        Initialize debouncer.
        
        Args:
            debounce_delay: Minimum time (seconds) between repeated gestures
        """
        self.debounce_delay = debounce_delay
        self.last_gesture_time = {}
    
    def should_trigger(self, gesture_name: str, current_time: float) -> bool:
        """
        Check if a gesture should trigger (not still debouncing).
        
        Args:
            gesture_name: Name of the gesture
            current_time: Current timestamp
            
        Returns:
            True if enough time has passed since last trigger
        """
        if gesture_name not in self.last_gesture_time:
            self.last_gesture_time[gesture_name] = current_time
            return True
        
        time_since_last = current_time - self.last_gesture_time[gesture_name]
        
        if time_since_last >= self.debounce_delay:
            self.last_gesture_time[gesture_name] = current_time
            return True
        
        return False
    
    def reset(self, gesture_name: Optional[str] = None):
        """Reset debounce timer for a gesture or all gestures."""
        if gesture_name is None:
            self.last_gesture_time.clear()
        else:
            self.last_gesture_time.pop(gesture_name, None)


class MotionTracker:
    """Track motion across frames for swipe detection."""
    
    def __init__(self, history_size: int = 10):
        """
        Initialize motion tracker.
        
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = history_size
        self.position_history: deque = deque(maxlen=history_size)
        self.time_history: deque = deque(maxlen=history_size)
    
    def add_position(self, point: Point, timestamp: float):
        """Add a position to the history."""
        self.position_history.append(point)
        self.time_history.append(timestamp)
    
    def get_displacement(self) -> Optional[Tuple[float, float]]:
        """
        Get displacement from oldest to newest position.
        
        Returns:
            (dx, dy) displacement or None if insufficient history
        """
        if len(self.position_history) < 2:
            return None
        
        oldest = self.position_history[0]
        newest = self.position_history[-1]
        
        return (newest.x - oldest.x, newest.y - oldest.y)
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Get velocity (displacement per second).
        
        Returns:
            (vx, vy) velocity or None if insufficient history
        """
        if len(self.time_history) < 2:
            return None
        
        displacement = self.get_displacement()
        if displacement is None:
            return None
        
        time_delta = self.time_history[-1] - self.time_history[0]
        if time_delta == 0:
            return None
        
        return (displacement[0] / time_delta, displacement[1] / time_delta)
    
    def clear(self):
        """Clear motion history."""
        self.position_history.clear()
        self.time_history.clear()


def calculate_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return p1.distance_to(p2)


def calculate_angle(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.
    
    Returns angle in radians (0 to π).
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (mag1 * mag2)
    cos_angle = np.clip(cos_angle, -1, 1)  # Clamp to [-1, 1]
    
    return np.arccos(cos_angle)


def screen_coordinates(point: Point, frame_width: int, frame_height: int, 
                       screen_width: int, screen_height: int) -> Tuple[int, int]:
    """
    Convert normalized point coordinates to screen coordinates.
    
    Args:
        point: Point with normalized coordinates (0-1)
        frame_width: Video frame width
        frame_height: Video frame height
        screen_width: Screen width
        screen_height: Screen height
        
    Returns:
        (x, y) screen coordinates
    """
    screen_x = int(point.x * frame_width * (screen_width / frame_width))
    screen_y = int(point.y * frame_height * (screen_height / frame_height))
    
    # Clamp to screen bounds
    screen_x = max(0, min(screen_x, screen_width - 1))
    screen_y = max(0, min(screen_y, screen_height - 1))
    
    return (screen_x, screen_y)


def is_finger_up(finger_tip: Point, finger_pip: Point, threshold: float = 0.02) -> bool:
    """
    Determine if a finger is extended (up).
    
    A finger is considered "up" if its tip is higher (lower y value) than its PIP joint.
    
    Args:
        finger_tip: Position of finger tip
        finger_pip: Position of finger PIP joint
        threshold: Threshold for the comparison
        
    Returns:
        True if finger is extended
    """
    return finger_tip.y < (finger_pip.y - threshold)


def are_landmarks_visible(landmarks: List[Point], min_visibility: float = 0.5) -> bool:
    """Check if landmarks are visible/confident enough."""
    if not landmarks:
        return False
    
    # This is a placeholder - in actual implementation, 
    # landmarks would carry visibility scores
    return True
