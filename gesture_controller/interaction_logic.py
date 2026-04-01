"""
Interaction Logic Module
Manages interaction modes, gesture-to-action mapping, and system commands.
"""

import time
import pyautogui
from typing import Optional, Callable, Dict
from enum import Enum

from .config import GestureType, InteractionMode, GESTURE_ACTIONS, DetectionThresholds
from .hand_detector import HandLandmarks
from .utils import PointSmoother, Debouncer, Point


class InteractionLogic:
    """
    Manages the interaction between detected gestures and system actions.
    
    Features:
    - Gesture-to-command mapping
    - Interaction mode management
    - Cursor smoothing and stability
    - Command debouncing
    """
    
    def __init__(self, screen_width: int = 2560, screen_height: int = 1440,
                 thresholds: DetectionThresholds = None, target_fps: int = 30,
                 enable_actions: bool = True):
        """
        Initialize interaction logic.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            thresholds: Detection thresholds
            target_fps: Target frames per second
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.thresholds = thresholds or DetectionThresholds()
        self.target_fps = target_fps
        self.enable_actions = enable_actions
        
        # Current state
        self.current_mode = InteractionMode.IDLE
        self.current_gesture = GestureType.UNKNOWN
        self.last_gesture_time = None
        self.mode_start_time = None
        
        # Cursor tracking
        self.point_smoother = PointSmoother(alpha=self.thresholds.smoothing_alpha)
        self.current_cursor_pos = (0, 0)
        
        # Debouncing
        self.debouncer = Debouncer(self.thresholds.debounce_delay)
        
        # Frame timing
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_actual = 0
        
        # Action handlers
        self.action_handlers: Dict[str, Callable] = {
            "cursor_move": self._handle_cursor_move,
            "click": self._handle_click,
            "reset": self._handle_reset,
            "lock": self._handle_lock,
            "toggle_playpause": self._handle_toggle_playpause,
            "switch_app": self._handle_switch_app,
            "previous_tab": self._handle_previous_tab,
            "next_tab": self._handle_next_tab,
            "volume_up": self._handle_volume_up,
            "volume_down": self._handle_volume_down,
            "drag": self._handle_drag,
            "scroll": self._handle_scroll,
            "none": lambda *args, **kwargs: None,
        }
        
        # Drag state
        self.is_dragging = False
        self.drag_start_pos = None
    
    def update(self, gesture: GestureType, hand: HandLandmarks,
              current_time: Optional[float] = None) -> Dict:
        """
        Update interaction state based on detected gesture and hand.
        
        Args:
            gesture: Detected gesture type
            hand: Hand landmarks
            current_time: Current timestamp
            
        Returns:
            Dictionary with interaction status and performed actions
        """
        if current_time is None:
            current_time = time.time()
        
        # Update FPS
        self._update_fps(current_time)
        
        # Update current gesture
        self.current_gesture = gesture
        self.last_gesture_time = current_time
        
        # Handle mode transitions
        self._update_interaction_mode(gesture, current_time)
        
        # Handle gesture actions based on current mode
        status = self._handle_gesture_action(gesture, hand, current_time)
        
        return status
    
    def _update_interaction_mode(self, gesture: GestureType, current_time: float):
        """Update interaction mode based on gesture."""
        if gesture == GestureType.OPEN_PALM and self.current_mode == InteractionMode.IDLE:
            self.current_mode = InteractionMode.CONTROL
            self.mode_start_time = current_time
        elif gesture == GestureType.FIST and self.current_mode != InteractionMode.LOCKED:
            self.current_mode = InteractionMode.LOCKED
            self.mode_start_time = current_time
        elif gesture == GestureType.OPEN_PALM and self.current_mode == InteractionMode.LOCKED:
            self.current_mode = InteractionMode.IDLE
            self.mode_start_time = current_time
    
    def _handle_gesture_action(self, gesture: GestureType, hand: HandLandmarks,
                              current_time: float) -> Dict:
        """Execute action corresponding to gesture."""
        
        action = GESTURE_ACTIONS.get(gesture, "none")
        
        status = {
            "gesture": gesture.value,
            "mode": self.current_mode.value,
            "action": action,
            "executed": False,
            "cursor_pos": self.current_cursor_pos,
        }
        
        # Only execute in appropriate modes
        if self.current_mode == InteractionMode.LOCKED:
            if gesture != GestureType.OPEN_PALM:
                return status
        
        if not self.enable_actions:
            status["action"] = f"preview:{action}"
            return status

        if action in self.action_handlers:
            try:
                self.action_handlers[action](gesture, hand, current_time)
                status["executed"] = True
            except Exception as e:
                status["error"] = str(e)
        
        return status
    
    def _handle_cursor_move(self, gesture: GestureType, hand: HandLandmarks,
                           current_time: float):
        """Move cursor based on index finger position."""
        # Use index finger tip for cursor control
        index_pos = hand.index_finger_tip
        
        # Smooth the position
        smooth_pos = self.point_smoother.smooth(index_pos)
        
        # Convert to screen coordinates
        screen_x = int(smooth_pos.x * self.screen_width)
        screen_y = int(smooth_pos.y * self.screen_height)
        
        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        self.current_cursor_pos = (screen_x, screen_y)
        pyautogui.moveTo(screen_x, screen_y, duration=0.01)
    
    def _handle_click(self, gesture: GestureType, hand: HandLandmarks,
                     current_time: float):
        """Execute mouse click."""
        if self.debouncer.should_trigger("click", current_time):
            pyautogui.click()
    
    def _handle_reset(self, gesture: GestureType, hand: HandLandmarks,
                     current_time: float):
        """Reset to idle mode."""
        self.current_mode = InteractionMode.IDLE
        self.point_smoother.reset()
    
    def _handle_lock(self, gesture: GestureType, hand: HandLandmarks,
                    current_time: float):
        """Lock interaction."""
        if self.current_mode != InteractionMode.LOCKED:
            self.current_mode = InteractionMode.LOCKED
    
    def _handle_toggle_playpause(self, gesture: GestureType, hand: HandLandmarks,
                                current_time: float):
        """Toggle play/pause."""
        if self.debouncer.should_trigger("playpause", current_time):
            pyautogui.press('space')
    
    def _handle_switch_app(self, gesture: GestureType, hand: HandLandmarks,
                          current_time: float):
        """Switch between applications."""
        if self.debouncer.should_trigger("switch_app", current_time):
            pyautogui.hotkey('alt', 'tab')
    
    def _handle_previous_tab(self, gesture: GestureType, hand: HandLandmarks,
                            current_time: float):
        """Navigate to previous tab."""
        if self.debouncer.should_trigger("prev_tab", current_time):
            pyautogui.hotkey('ctrl', 'shift', 'tab')
    
    def _handle_next_tab(self, gesture: GestureType, hand: HandLandmarks,
                        current_time: float):
        """Navigate to next tab."""
        if self.debouncer.should_trigger("next_tab", current_time):
            pyautogui.hotkey('ctrl', 'tab')
    
    def _handle_volume_up(self, gesture: GestureType, hand: HandLandmarks,
                         current_time: float):
        """Increase volume."""
        if self.debouncer.should_trigger("vol_up", current_time):
            # Using keyboard shortcut (depends on OS)
            # For macOS: brightness up is Option + Shift + F2
            pyautogui.press('volumeup')
    
    def _handle_volume_down(self, gesture: GestureType, hand: HandLandmarks,
                           current_time: float):
        """Decrease volume."""
        if self.debouncer.should_trigger("vol_down", current_time):
            pyautogui.press('volumedown')
    
    def _handle_drag(self, gesture: GestureType, hand: HandLandmarks,
                    current_time: float):
        """Drag using pinch gesture."""
        if not self.is_dragging:
            # Start drag
            self.is_dragging = True
            self.drag_start_pos = hand.index_finger_tip
            pyautogui.mouseDown()
        else:
            # Continue drag
            self._handle_cursor_move(gesture, hand, current_time)
    
    def _handle_scroll(self, gesture: GestureType, hand: HandLandmarks,
                      current_time: float):
        """Scroll using palm movement."""
        # Use palm center for scroll direction
        palm_center = hand.get_center()
        
        # Simple scroll: if palm moves down, scroll down
        # This would need motion history for proper implementation
        if palm_center.y > 0.5:
            pyautogui.scroll(-3)
        else:
            pyautogui.scroll(3)
    
    def end_drag(self):
        """End drag operation."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
            self.drag_start_pos = None
    
    def _update_fps(self, current_time: float):
        """Update FPS counter."""
        self.frame_count += 1
        time_elapsed = current_time - self.last_frame_time
        
        if time_elapsed >= 1.0:
            self.fps_actual = self.frame_count / time_elapsed
            self.frame_count = 0
            self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps_actual
    
    def close(self):
        """Clean up resources."""
        self.end_drag()
        self.point_smoother.reset()
