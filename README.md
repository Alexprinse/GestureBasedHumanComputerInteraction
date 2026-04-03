# Intelligent Gesture-Based Human-Computer Interaction System

A real-time webcam-based gesture control system built with MediaPipe, OpenCV, and PyAutoGUI. It detects hand landmarks, classifies gestures, and turns them into computer interactions such as cursor movement, click-and-drag selection, media control, scrolling, and app/tab switching.

## Overview

The current version is optimized for practical laptop use and presentation. It uses an always-on interaction model with smoother cursor motion, calmer gesture triggering, and gesture-specific directional controls.

## Current Gesture Set

### Core Gestures

- `index_finger_up` - move cursor
- `pinch` - click or drag selection when moved
- `open_palm` - scroll with palm movement
- `fist` - directional control for volume and tab switching
- `two_fingers` - play/pause
- `three_fingers` - switch application

### Motion Gestures

- `swipe_left` - previous tab
- `swipe_right` - next tab
- `swipe_up` - volume up
- `swipe_down` - volume down

### Disabled or Not Used in Current Flow

- `side_horizontal`
- `side_vertical`

These are kept in the codebase for compatibility, but the current interaction flow does not rely on them.

## How the Current System Works

1. The webcam captures the frame.
2. MediaPipe Hand Landmarker extracts 21 hand landmarks.
3. The gesture recognizer classifies the hand pose.
4. The interaction layer maps the gesture to a system action.
5. PyAutoGUI or macOS automation executes the action.

## Functional Behavior

### Cursor and Selection

- `index_finger_up` moves the cursor.
- `pinch` starts selection behavior.
- If you pinch and move, it becomes click-and-drag for text selection.
- Releasing the pinch ends the drag.

### Volume and Tab Control

- `fist` uses thumb direction as the control signal.
- Thumb up/down controls volume up/down.
- Thumb left/right controls previous/next tab.
- This replaced side-horizontal interaction because side pose could be confused with open palm.

### App Switching

- `three_fingers` switches applications.
- On macOS, this uses `Command + Tab`.

### Media Control

- `two_fingers` toggles play/pause.

### Scrolling

- `open_palm` is used for smooth vertical scrolling.
- Motion is intentionally slowed and stabilized for better presentation.

## Processing Pipeline

```text
Webcam Frame
  -> MediaPipe Hand Landmark Detection
  -> Gesture Recognition
  -> Temporal Stabilization
  -> Interaction Logic
  -> System Action via PyAutoGUI / macOS automation
```

## Project Structure

```text
majorProject/
├── gesture_controller/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── hand_detector.py
│   ├── gesture_recognizer.py
│   └── interaction_logic.py
├── main.py
├── models/
│   └── hand_landmarker.task
└── README.md
```

## Key Modules

### `gesture_controller/hand_detector.py`

- Runs MediaPipe hand landmark detection
- Returns real 21-point hand landmarks
- Supports visualization of landmarks and connections

### `gesture_controller/gesture_recognizer.py`

- Classifies hand gestures from landmarks
- Uses finger-state rules and geometric checks
- Stabilizes output across frames to reduce flicker
- Keeps side gestures disabled in the current interaction flow

### `gesture_controller/interaction_logic.py`

- Maps gestures to actions
- Handles cursor motion, click, drag, scroll, volume, tabs, and app switching
- Uses smoothing and debouncing to keep the interaction calm and presentable
- Supports macOS-friendly shortcuts

### `gesture_controller/config.py`

- Defines gesture symbols
- Defines thresholds and smoothing values
- Stores the gesture-to-action mapping

### `gesture_controller/utils.py`

- Point and distance helpers
- Exponential moving average smoothing
- Motion tracking
- Debouncing utilities

## Important Symbols

### Gesture Symbols

- `GestureType.INDEX_FINGER_UP`
- `GestureType.PINCH`
- `GestureType.OPEN_PALM`
- `GestureType.FIST`
- `GestureType.TWO_FINGERS`
- `GestureType.THREE_FINGERS`
- `GestureType.SWIPE_LEFT`
- `GestureType.SWIPE_RIGHT`
- `GestureType.SWIPE_UP`
- `GestureType.SWIPE_DOWN`
- `GestureType.SIDE_HORIZONTAL`
- `GestureType.SIDE_VERTICAL`
- `GestureType.UNKNOWN`

### Action Symbols

- `cursor_move`
- `click`
- `drag`
- `scroll`
- `toggle_playpause`
- `switch_app`
- `previous_tab`
- `next_tab`
- `volume_up`
- `volume_down`
- `none`

## Running the App

```bash
python main.py
```

Useful options:

```bash
python main.py --debug
python main.py --test-mode --debug
python main.py --camera 0
python main.py --screen-width 1920 --screen-height 1080
```

## Recommended Workflow

1. Run `python main.py --test-mode --debug` first.
2. Check the on-screen gesture label and cursor position.
3. Test one gesture at a time.
4. Move to normal mode once the gesture behavior looks correct.

## Tuning Notes

The current interaction is intentionally slowed down for smoother presentation.

Main tuning values are in `gesture_controller/config.py` and `gesture_controller/interaction_logic.py`:

- `smoothing_alpha` controls how much hand movement is filtered
- `debounce_delay` controls how fast repeated actions can fire
- `cursor_lerp_alpha` controls cursor glide speed
- `cursor_deadzone_px` reduces small cursor jumps
- `pinch_drag_threshold` controls how much movement is needed before selection drag starts

## macOS Notes

This project is currently tuned for macOS use.

- App switching uses `Command + Tab`
- Previous tab uses `Command + Shift + [`
- Next tab uses `Command + Shift + ]`
- Volume changes use AppleScript-based output volume control

## Dependencies

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## Troubleshooting

### Gesture feels too fast

- Lower `cursor_lerp_alpha`
- Increase `debounce_delay`
- Increase `pinch_drag_threshold`

### Volume or tab switching does not trigger

- Make sure you are holding a fist
- Point the thumb clearly in one direction (up, down, left, or right)
- Use `--debug` to watch the detected gesture name

### Cursor looks unstable

- Improve lighting
- Keep your hand in frame
- Reduce background clutter

### macOS permissions

- Grant camera access
- Grant accessibility permissions for keyboard and mouse control

## Status

- Version: 1.0.0
- Last Updated: April 2026
- Status: Active Development
