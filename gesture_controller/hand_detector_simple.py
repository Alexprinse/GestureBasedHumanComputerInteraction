"""
Simplified Hand Detection Module
Uses OpenCV-based hand detection with landmark simulation for gesture recognition.
This is a fallback when MediaPipe models aren't available.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .utils import Point
from .config import DetectionThresholds


@dataclass
class HandLandmarks:
    """Container for hand landmark data."""
    landmarks: List[Point]  # 21 landmarks
    wrist: Point
    index_finger_tip: Point
    middle_finger_tip: Point
    ring_finger_tip: Point
    pinky_tip: Point
    thumb_tip: Point
    thumb_ip: Point
    thumb_pip: Point
    handedness: str  # "Left" or "Right"
    confidence: float
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x_min, y_min, x_max, y_max) in normalized coordinates."""
        x_coords = [lm.x for lm in self.landmarks]
        y_coords = [lm.y for lm in self.landmarks]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_center(self) -> Point:
        """Get center of the hand."""
        x = sum(lm.x for lm in self.landmarks) / len(self.landmarks)
        y = sum(lm.y for lm in self.landmarks) / len(self.landmarks)
        z = sum(lm.z for lm in self.landmarks) / len(self.landmarks)
        
        return Point(x, y, z)


class HandDetector:
    """
    Simplified hand detector using skin color detection and contour analysis.
    Falls back from MediaPipe when models unavailable.
    """
    
    # Landmark indices
    WRIST = 0
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_PIP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    
    def __init__(self, thresholds: DetectionThresholds = None):
        """
        Initialize the simplified hand detector.
        
        Args:
            thresholds: Detection thresholds configuration
        """
        self.thresholds = thresholds or DetectionThresholds()
        self.previous_hand = None
        
        # Skin color detection parameters (HSV range for skin)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    def detect(self, frame: np.ndarray) -> Tuple[List[HandLandmarks], np.ndarray]:
        """
        Detect hands using skin color and contour analysis.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (list of HandLandmarks, annotated frame)
        """
        hands_data = []
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < 5000:  # Adjust based on camera and distance
                continue
            
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Get convex hull for hand
            hull = cv2.convexHull(contour)
            
            # Generate simplified landmarks from contour
            landmarks = self._generate_landmarks_from_contour(
                contour, hull, x, y, cw, ch, w, h
            )
            
            if landmarks and len(landmarks) >= 21:
                # Create HandLandmarks object with generated landmarks
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
                    handedness="Right",  # Simplified - always assume right
                    confidence=0.7
                )
                
                hands_data.append(hand)
        
        return hands_data, annotated_frame
    
    def _generate_landmarks_from_contour(self, contour, hull, x, y, w, h, 
                                        frame_w, frame_h) -> List[Point]:
        """
        Generate 21 hand landmarks from contour and hull.
        
        This is a simplified approximation when MediaPipe isn't available.
        """
        landmarks = []
        
        # Normalize contour coordinates
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.contourArea(contour), True)
        
        # Ensure we have at least 21 points
        while len(approx) < 21:
            approx = np.vstack([approx, approx[-1:]])
        
        # Sample 21 evenly distributed points along the contour
        indices = np.linspace(0, len(approx) - 1, 21, dtype=int)
        
        for idx in indices:
            pt = approx[idx][0]
            norm_x = (pt[0] + x) / frame_w
            norm_y = (pt[1] + y) / frame_h
            
            # Clamp to [0, 1]
            norm_x = max(0, min(1, norm_x))
            norm_y = max(0, min(1, norm_y))
            
            landmarks.append(Point(norm_x, norm_y, 0.0))
        
        return landmarks
    
    def draw_landmarks(self, frame: np.ndarray, hands: List[HandLandmarks], 
                      draw_connections: bool = True, 
                      draw_labels: bool = False) -> np.ndarray:
        """
        Draw simplified hand landmarks on frame.
        
        Args:
            frame: Input frame
            hands: List of detected hands
            draw_connections: Whether to draw skeleton connections
            draw_labels: Whether to draw landmark labels
            
        Returns:
            Frame with drawn landmarks
        """
        h, w, c = frame.shape
        annotated = frame.copy()
        
        for hand in hands:
            # Draw landmarks as circles
            for i, landmark in enumerate(hand.landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Use different colors for different landmark types
                if i == 0:  # Wrist
                    color = (0, 255, 0)
                    radius = 6
                elif i in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 0, 255)
                    radius = 5
                else:
                    color = (255, 0, 0)
                    radius = 3
                
                cv2.circle(annotated, (x, y), radius, color, -1)
                
                if draw_labels:
                    cv2.putText(annotated, str(i), (x + 5, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw bounding box
            bbox = hand.get_bounding_box()
            x_min = int(bbox[0] * w)
            y_min = int(bbox[1] * h)
            x_max = int(bbox[2] * w)
            y_max = int(bbox[3] * h)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw label
            label = f"{hand.handedness} ({hand.confidence:.2f})"
            cv2.putText(annotated, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated
    
    def close(self):
        """Clean up resources."""
        pass
