"""
Gesture Controller Package
A real-time, vision-based interface for gesture-based human-computer interaction.
"""

__version__ = "1.0.0"
__author__ = "Gesture Control Team"

from .hand_detector import HandDetector
from .gesture_recognizer import GestureRecognizer
from .interaction_logic import InteractionLogic

__all__ = ["HandDetector", "GestureRecognizer", "InteractionLogic"]
