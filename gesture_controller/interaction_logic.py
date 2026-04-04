"""
Interaction Logic Module
Manages interaction modes, gesture-to-action mapping, and system commands.
"""

import time
import sys
import subprocess
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
        
        # Current mode state.
        self.current_mode = InteractionMode.CONTROL
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
            "zoom_in": self._handle_zoom_in,
            "zoom_out": self._handle_zoom_out,
            "volume_up": self._handle_volume_up,
            "volume_down": self._handle_volume_down,
            "drag": self._handle_drag,
            "scroll": self._handle_scroll,
            "none": lambda *args, **kwargs: None,
        }
        
        # Drag state
        self.is_dragging = False
        self.drag_start_pos = None

        # Cursor smoothing controls (slower, more stable movement).
        self.cursor_lerp_alpha = 0.18
        self.cursor_deadzone_px = 4

        # Fractional scroll carry-over for smoother palm scrolling.
        self.scroll_accumulator: float = 0.0

        # Palm motion state for smooth scrolling and vertical action gating.
        self.prev_palm_center: Optional[Point] = None
        self.prev_palm_time: Optional[float] = None
        self.prev_side_center: Optional[Point] = None
        self.prev_side_time: Optional[float] = None

        # Pinch-drag state for text selection like click-and-drag.
        self.pinch_start_tip: Optional[Point] = None
        self.pinch_start_time: Optional[float] = None
        self.pinch_drag_threshold: float = 0.022

        # Hold after any tab switch to avoid rapid tab hopping.
        self.tab_switch_cooldown_seconds: float = 0.35
        self.tab_switch_cooldown_until: float = 0.0

        # Separate cooldown for three-finger app switching.
        self.switch_app_cooldown_seconds: float = 0.45
        self.switch_app_cooldown_until: float = 0.0

        # Fast click gating for pinch release click.
        self.click_cooldown_seconds: float = 0.18
        self.click_cooldown_until: float = 0.0

        # Hold after play/pause toggle (two-fingers) to avoid fast retriggering.
        self.playpause_cooldown_seconds: float = 1.0
        self.playpause_cooldown_until: float = 0.0

        # Hold after side-horizontal zoom action.
        self.side_zoom_cooldown_seconds: float = 0.85
        self.side_zoom_cooldown_until: float = 0.0

    
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
        previous_gesture = self.current_gesture
        self.current_gesture = gesture
        self.last_gesture_time = current_time
        
        # Handle mode transitions
        self._update_interaction_mode(gesture, current_time)
        
        # Handle gesture actions based on current mode
        status = self._handle_gesture_action(gesture, hand, current_time, previous_gesture)
        
        return status
    
    def _update_interaction_mode(self, gesture: GestureType, current_time: float):
        """Update mode: fist locks controls, open palm unlocks."""
        if gesture == GestureType.FIST:
            self.current_mode = InteractionMode.LOCKED
            return

        if self.current_mode == InteractionMode.LOCKED and gesture == GestureType.OPEN_PALM:
            self.current_mode = InteractionMode.CONTROL
    
    def _handle_gesture_action(self, gesture: GestureType, hand: HandLandmarks,
                              current_time: float, previous_gesture: GestureType) -> Dict:
        """Execute action corresponding to gesture."""
        
        action = GESTURE_ACTIONS.get(gesture, "none")
        
        status = {
            "gesture": gesture.value,
            "mode": self.current_mode.value,
            "action": action,
            "executed": False,
            "cursor_pos": self.current_cursor_pos,
        }

        # In locked mode, stop all implementations/actions.
        if self.current_mode == InteractionMode.LOCKED:
            status["action"] = "lock"
            self.prev_side_center = None
            self.prev_side_time = None
            self.pinch_start_tip = None
            self.pinch_start_time = None
            if self.is_dragging:
                self.end_drag()
            return status

        # Handle pinch release first (end drag or click fallback).
        if previous_gesture == GestureType.PINCH and gesture != GestureType.PINCH:
            if self.is_dragging:
                self.end_drag()
                status["action"] = "drag_end"
                status["executed"] = True
            else:
                if self.enable_actions:
                    self._handle_click(GestureType.PINCH, hand, current_time)
                    status["executed"] = True
                status["action"] = "click"
            self.pinch_start_tip = None
            self.pinch_start_time = None

        # Open palm acts as smooth scroll by palm vertical motion.
        if gesture == GestureType.OPEN_PALM:
            action = "scroll"
            status["action"] = action
            self.prev_side_center = None
            self.prev_side_time = None

        # Side-horizontal: vertical motion controls zoom.
        # - Up: zoom in
        # - Down: zoom out
        if gesture == GestureType.SIDE_HORIZONTAL:
            self.pinch_start_tip = None
            self.pinch_start_time = None
            side_action = self._detect_side_horizontal_zoom_action(hand, current_time)

            if side_action in ("zoom_in", "zoom_out") and current_time < self.side_zoom_cooldown_until:
                side_action = None

            status["action"] = side_action if side_action else "none"
            if not self.enable_actions:
                status["action"] = f"preview:{status['action']}"
                return status
            if side_action and side_action in self.action_handlers:
                try:
                    self.action_handlers[side_action](gesture, hand, current_time)
                    status["executed"] = True
                    if side_action in ("zoom_in", "zoom_out"):
                        self.side_zoom_cooldown_until = current_time + self.side_zoom_cooldown_seconds
                except Exception as e:
                    status["error"] = str(e)
            return status

        # side_vertical disabled for now.
        if gesture == GestureType.SIDE_VERTICAL:
            action = "none"
            status["action"] = action if self.enable_actions else f"preview:{action}"
            return status

        # Pinch gesture supports click-and-drag selection.
        if gesture == GestureType.PINCH:
            self._handle_cursor_move(gesture, hand, current_time)
            tip = hand.index_finger_tip

            if self.pinch_start_tip is None:
                self.pinch_start_tip = Point(tip.x, tip.y)
                self.pinch_start_time = current_time
                status["action"] = "pinch_hold"
                return status

            moved = self.pinch_start_tip.distance_2d(Point(tip.x, tip.y))
            if not self.is_dragging and moved >= self.pinch_drag_threshold:
                if self.enable_actions:
                    pyautogui.mouseDown()
                self.is_dragging = True
                self.drag_start_pos = tip

            if self.is_dragging:
                if self.enable_actions:
                    self._handle_cursor_move(gesture, hand, current_time)
                status["action"] = "drag"
                status["executed"] = self.enable_actions
            else:
                status["action"] = "pinch_hold"
            return status

        # Reset side motion history when not in side-horizontal gesture.
        self.prev_side_center = None
        self.prev_side_time = None
        
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
        
        curr_x, curr_y = self.current_cursor_pos
        if curr_x == 0 and curr_y == 0:
            new_x, new_y = screen_x, screen_y
        else:
            new_x = int((1.0 - self.cursor_lerp_alpha) * curr_x + self.cursor_lerp_alpha * screen_x)
            new_y = int((1.0 - self.cursor_lerp_alpha) * curr_y + self.cursor_lerp_alpha * screen_y)

        if abs(new_x - curr_x) < self.cursor_deadzone_px and abs(new_y - curr_y) < self.cursor_deadzone_px:
            return

        self.current_cursor_pos = (new_x, new_y)
        pyautogui.moveTo(new_x, new_y, duration=0)
    
    def _handle_click(self, gesture: GestureType, hand: HandLandmarks,
                     current_time: float):
        """Execute mouse click."""
        if current_time < self.click_cooldown_until:
            return
        pyautogui.click()
        self.click_cooldown_until = current_time + self.click_cooldown_seconds
    
    def _handle_reset(self, gesture: GestureType, hand: HandLandmarks,
                     current_time: float):
        """No-op in always-on control mode."""
        return
    
    def _handle_lock(self, gesture: GestureType, hand: HandLandmarks,
                    current_time: float):
        """No-op in always-on control mode."""
        return
    
    def _handle_toggle_playpause(self, gesture: GestureType, hand: HandLandmarks,
                                current_time: float):
        """Toggle play/pause."""
        if current_time < self.playpause_cooldown_until:
            return

        if self.debouncer.should_trigger("playpause", current_time):
            pyautogui.press('space')
            self.playpause_cooldown_until = current_time + self.playpause_cooldown_seconds
    
    def _handle_switch_app(self, gesture: GestureType, hand: HandLandmarks,
                          current_time: float):
        """Switch between applications."""
        if current_time < self.switch_app_cooldown_until:
            return

        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'tab')
        else:
            pyautogui.hotkey('alt', 'tab')

        self.switch_app_cooldown_until = current_time + self.switch_app_cooldown_seconds
    
    def _handle_previous_tab(self, gesture: GestureType, hand: HandLandmarks,
                            current_time: float):
        """Navigate to previous tab."""
        if current_time < self.tab_switch_cooldown_until:
            return

        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'shift', '[')
        else:
            pyautogui.hotkey('ctrl', 'shift', 'tab')
        self.tab_switch_cooldown_until = current_time + self.tab_switch_cooldown_seconds
    
    def _handle_next_tab(self, gesture: GestureType, hand: HandLandmarks,
                        current_time: float):
        """Navigate to next tab."""
        if current_time < self.tab_switch_cooldown_until:
            return

        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'shift', ']')
        else:
            pyautogui.hotkey('ctrl', 'tab')
        self.tab_switch_cooldown_until = current_time + self.tab_switch_cooldown_seconds
    
    def _handle_volume_up(self, gesture: GestureType, hand: HandLandmarks,
                         current_time: float):
        """Increase volume."""
        if self.debouncer.should_trigger("vol_up", current_time):
            self._adjust_system_volume(6)
    
    def _handle_volume_down(self, gesture: GestureType, hand: HandLandmarks,
                           current_time: float):
        """Decrease volume."""
        if self.debouncer.should_trigger("vol_down", current_time):
            self._adjust_system_volume(-6)

    def _handle_zoom_in(self, gesture: GestureType, hand: HandLandmarks,
                       current_time: float):
        """Zoom in (app-level)."""
        if self.debouncer.should_trigger("zoom_in", current_time):
            if sys.platform == "darwin":
                pyautogui.hotkey('command', '+')
            else:
                pyautogui.hotkey('ctrl', '+')

    def _handle_zoom_out(self, gesture: GestureType, hand: HandLandmarks,
                        current_time: float):
        """Zoom out (app-level)."""
        if self.debouncer.should_trigger("zoom_out", current_time):
            if sys.platform == "darwin":
                pyautogui.hotkey('command', '-')
            else:
                pyautogui.hotkey('ctrl', '-')

    def _adjust_system_volume(self, delta: int):
        """Adjust system output volume with macOS-friendly fallback."""
        if sys.platform == "darwin":
            # Clamp output volume to [0, 100].
            script = (
                f"set v to output volume of (get volume settings)\n"
                f"set v to v + ({delta})\n"
                "if v > 100 then set v to 100\n"
                "if v < 0 then set v to 0\n"
                "set volume output volume v"
            )
            subprocess.run(["osascript", "-e", script], check=False)
            return

        pyautogui.press('volumeup' if delta > 0 else 'volumedown')

    def _detect_side_horizontal_zoom_action(self, hand: HandLandmarks,
                                           current_time: float) -> Optional[str]:
        """Map side-horizontal vertical movement to zoom action names."""
        center = hand.get_center()

        if self.prev_side_center is None or self.prev_side_time is None:
            self.prev_side_center = center
            self.prev_side_time = current_time
            return None

        dx = center.x - self.prev_side_center.x
        dy = center.y - self.prev_side_center.y

        self.prev_side_center = center
        self.prev_side_time = current_time

        # Ignore micro movement noise.
        if abs(dx) < 0.014 and abs(dy) < 0.014:
            return None

        if abs(dy) > abs(dx) * 1.15:
            return "zoom_in" if dy < 0 else "zoom_out"

        return None

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
        """Smooth scroll based on open-palm vertical movement (dy)."""
        palm_center = hand.get_center()

        if self.prev_palm_center is None or self.prev_palm_time is None:
            self.prev_palm_center = palm_center
            self.prev_palm_time = current_time
            return

        dt = max(current_time - self.prev_palm_time, 1e-6)
        dy = palm_center.y - self.prev_palm_center.y
        vy = dy / dt

        # Ignore micro-jitter.
        if abs(dy) < 0.004 and abs(vy) < 0.12:
            self.scroll_accumulator *= 0.85
            self.prev_palm_center = palm_center
            self.prev_palm_time = current_time
            return

        # Convert normalized motion to fractional scroll and emit integral steps.
        self.scroll_accumulator += -dy * 180
        steps = int(max(-9, min(9, self.scroll_accumulator)))
        if steps != 0:
            pyautogui.scroll(steps)
            self.scroll_accumulator -= steps

        self.prev_palm_center = palm_center
        self.prev_palm_time = current_time

    def _is_sideward_palm(self, hand: HandLandmarks) -> bool:
        """Detect sideward palm orientation from knuckle span compression."""
        # When palm turns sideward to camera, knuckle span appears compressed.
        knuckle_span = hand.landmarks[5].distance_2d(hand.landmarks[17])
        palm_length = hand.landmarks[0].distance_2d(hand.landmarks[9])
        if palm_length <= 1e-6:
            return False
        ratio = knuckle_span / palm_length
        return ratio < 1.15

    
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
        self.scroll_accumulator = 0.0
        self.prev_palm_center = None
        self.prev_palm_time = None
        self.prev_side_center = None
        self.prev_side_time = None
