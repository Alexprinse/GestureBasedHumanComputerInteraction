#!/usr/bin/env python3
"""Quick test of gesture controller components."""

if __name__ == "__main__":
    try:
        from gesture_controller import HandDetector, GestureRecognizer
        print("✓ Import successful!")
        
        # Try creating detector
        detector = HandDetector()
        print("✓ HandDetector initialized!")
        
        # Try creating recognizer
        recognizer = GestureRecognizer()
        print("✓ GestureRecognizer initialized!")
        
        print("\n✓ All components ready!")
        print("✓ You can now run: python main.py")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
