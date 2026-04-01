"""
Main Application
Intelligent Gesture-Based Human-Computer Interaction System

This is the entry point for the gesture control system. It captureslive video
from the webcam, detects hand gestures, and executes corresponding system commands.
"""

import cv2
import time
import argparse
from typing import Optional

from gesture_controller import HandDetector, GestureRecognizer
from gesture_controller.interaction_logic import InteractionLogic
from gesture_controller.config import (
    GestureType, InteractionMode, DetectionThresholds,
    DEFAULT_SYSTEM_CONFIG, DEFAULT_THRESHOLDS, DEFAULT_SCREEN_MAPPING
)


class GestureControlApp:
    """Main application class for gesture-based control."""
    
    def __init__(self, camera_id: int = 0, debug_mode: bool = False,
                 test_mode: bool = False):
        """
        Initialize the application.
        
        Args:
            camera_id: Webcam device ID (usually 0 for built-in)
            debug_mode: Enable debug visualization
        """
        self.debug_mode = debug_mode
        self.test_mode = test_mode
        self.running = False
        
        # Initialize components
        self.hand_detector = HandDetector(DEFAULT_THRESHOLDS)
        if self.hand_detector.profile_loaded:
            print(f"Detector profile loaded: {self.hand_detector.get_profile_path()}")
        else:
            print(f"Detector profile not found: {self.hand_detector.get_profile_path()}")
        self.gesture_recognizer = GestureRecognizer(DEFAULT_THRESHOLDS)
        self.interaction_logic = InteractionLogic(
            screen_width=DEFAULT_SCREEN_MAPPING.screen_width,
            screen_height=DEFAULT_SCREEN_MAPPING.screen_height,
            thresholds=DEFAULT_THRESHOLDS,
            target_fps=DEFAULT_SYSTEM_CONFIG.target_fps,
            enable_actions=not self.test_mode,
        )
        
        # Video capture
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Display settings
        self.show_fps = DEFAULT_SYSTEM_CONFIG.show_fps
        self.show_landmarks = DEFAULT_SYSTEM_CONFIG.show_landmarks
        self.show_gesture_label = DEFAULT_SYSTEM_CONFIG.show_gesture_label
        self.flip_frame = DEFAULT_SYSTEM_CONFIG.flip_frame
        
        # Statistics
        self.frame_count = 0
        self.gesture_counts = {}
        self.start_time = time.time()
    
    def run(self):
        """Main application loop."""
        print("Starting Gesture Control System...")
        print("Press 'q' to quit, 'r' to reset, 'd' to toggle debug info")
        if self.test_mode:
            print("TEST MODE enabled: system actions are disabled")
        print("-" * 60)
        
        self.running = True
        previous_hand = None
        hand_id_counter = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break
                
                frame_time = time.time()
                
                # Flip frame for mirror effect
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                
                # Detect hands
                hands, _ = self.hand_detector.detect(frame)
                
                # Process detected hands
                for hand_idx, hand in enumerate(hands):
                    hand_id = hand_idx
                    
                    # Recognize gesture
                    gesture, details = self.gesture_recognizer.recognize(
                        hand, hand_id, frame_time
                    )
                    
                    # Update interaction logic
                    status = self.interaction_logic.update(gesture, hand, frame_time)
                    
                    # Track gestures for statistics
                    if gesture != GestureType.UNKNOWN:
                        gesture_name = gesture.value
                        self.gesture_counts[gesture_name] = \
                            self.gesture_counts.get(gesture_name, 0) + 1
                    
                    # Debug output
                    if self.debug_mode:
                        print(f"Hand {hand_id}: Gesture={gesture.value}, "
                              f"Mode={status['mode']}, Action={status['action']}")
                
                # Draw annotations on frame
                frame = self._draw_annotations(frame, hands)
                
                # Display frame
                cv2.imshow("Gesture Control System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_system()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self._cleanup()
    
    def _draw_annotations(self, frame, hands):
        """Draw hand landmarks and gesture info on frame."""
        h, w, c = frame.shape
        
        # Draw hand landmarks
        if self.show_landmarks:
            frame = self.hand_detector.draw_landmarks(
                frame, hands, draw_connections=True, draw_labels=self.debug_mode
            )
        
        # Draw gesture and mode information
        if self.show_gesture_label:
            mode = self.interaction_logic.current_mode.value
            gesture = self.interaction_logic.current_gesture.value
            cursor_pos = self.interaction_logic.current_cursor_pos
            
            # Draw mode and gesture
            cv2.putText(frame, f"Mode: {mode}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Cursor: {cursor_pos}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw FPS
        if self.show_fps:
            fps = self.interaction_logic.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _reset_system(self):
        """Reset system state."""
        print("Resetting system...")
        self.gesture_recognizer.reset()
        self.interaction_logic = InteractionLogic(
            screen_width=DEFAULT_SCREEN_MAPPING.screen_width,
            screen_height=DEFAULT_SCREEN_MAPPING.screen_height,
            thresholds=DEFAULT_THRESHOLDS,
        )
    
    def _cleanup(self):
        """Clean up resources."""
        print("\nShutting down...")
        self.running = False
        self.cap.release()
        self.hand_detector.close()
        self.interaction_logic.close()
        cv2.destroyAllWindows()
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print session statistics."""
        elapsed_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Session duration: {elapsed_time:.1f} seconds")
        print(f"Average FPS: {self.frame_count / elapsed_time:.1f}")
        
        if self.gesture_counts:
            print("\nGesture Detection Summary:")
            for gesture, count in sorted(self.gesture_counts.items(),
                                        key=lambda x: x[1], reverse=True):
                print(f"  {gesture}: {count}")
        print("=" * 60)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Gesture-Based Human-Computer Interaction System"
    )
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed output")
    parser.add_argument("--test-mode", action="store_true",
                       help="Disable system actions for safe detection testing")
    parser.add_argument("--screen-width", type=int, default=2560,
                       help="Screen width in pixels (default: 2560)")
    parser.add_argument("--screen-height", type=int, default=1440,
                       help="Screen height in pixels (default: 1440)")
    
    args = parser.parse_args()
    
    # Update screen dimensions if provided
    if args.screen_width != 2560 or args.screen_height != 1440:
        DEFAULT_SCREEN_MAPPING.screen_width = args.screen_width
        DEFAULT_SCREEN_MAPPING.screen_height = args.screen_height
    
    # Create and run application
    app = GestureControlApp(
        camera_id=args.camera,
        debug_mode=args.debug,
        test_mode=args.test_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
