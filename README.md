# Intelligent Gesture-Based Human-Computer Interaction System

A real-time, vision-based interface that enables users to control a computer using hand gestures captured through a webcam. This project bridges human-computer interaction and robotics by treating gestures as control signals and the computer as an intelligent agent.

## 📋 Project Overview

The system leverages computer vision and machine learning techniques to detect, interpret, and map human hand gestures into meaningful system-level commands. It transforms the laptop into an interactive robotic entity, where gestures act as control signals, similar to how robots interpret sensory inputs for decision-making and actuation.

### Key Features

- ✨ **Real-time Hand Tracking**: MediaPipe detects 21 hand landmarks at 30+ FPS
- 👆 **Gesture Recognition**: Rule-based classification of static and motion gestures
- 🖱️ **System Control**: Mouse cursor movement, clicks, scrolling, and application control
- 🎯 **Temporal Smoothing**: Exponential moving average filters reduce jitter
- 🛡️ **Debouncing**: Prevents false positives and rapid repeated actions
- 📊 **Interactive Visualization**: Real-time feedback with FPS, gesture labels, and landmarks

## 🏗️ System Architecture

```
Input Layer
    ↓
Webcam → Video Stream
    ↓
Perception Layer
    ↓
MediaPipe Hand Detection (21 landmarks)
    ↓
Feature Extraction Layer
    ↓
Finger states, distances, motion vectors
    ↓
Gesture Recognition Layer
    ↓
Rule-based static & motion gesture classification
    ↓
Interaction Logic Layer
    ↓
Mode management, gesture-to-command mapping
    ↓
Execution Layer
    ↓
PyAutoGUI → System commands
```

## 🎨 Gesture Language

### Navigation Gestures
- **Index Finger Up**: Cursor Movement
- **Pinch** (Thumb + Index): Click / Drag

### Control Gestures
- **Open Palm**: Enter Control Mode / Reset
- **Fist**: Lock / Pause
- **Two Fingers** (Index + Middle): Play/Pause
- **Three Fingers** (Index + Middle + Ring): Switch Applications

### Motion-Based Gestures
- **Swipe Left**: Previous Tab
- **Swipe Right**: Next Tab
- **Swipe Up**: Volume Up
- **Swipe Down**: Volume Down

### Continuous Gestures
- **Pinch + Move**: Drag
- **Palm Move**: Scroll

## 📦 Project Structure

```
majorProject/
├── gesture_controller/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration & gesture definitions
│   ├── utils.py                    # Mathematical utilities & helpers
│   ├── hand_detector.py            # MediaPipe hand detection
│   ├── gesture_recognizer.py       # Gesture recognition engine
│   └── interaction_logic.py        # System interaction & command execution
├── tests/
│   └── __init__.py                 # Test package placeholder
├── main.py                         # Main application entry point
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- macOS, Linux, or Windows

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd /Users/shalem/Desktop/majorProject
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python main.py
```

**Command-line Options:**
```bash
python main.py --help
python main.py --camera 0 --debug  # Enable debug mode
python main.py --screen-width 1920 --screen-height 1080  # Set screen dimensions
```

### Using the System

1. **Launch the application**: `python main.py`
2. **Allow camera access** when prompted
3. **Position your hand** in front of the webcam
4. **Use gestures** to control your computer:
   - Open palm to enter control mode
   - Index finger up for cursor
   - Pinch to click
   - Fist to lock interaction
5. **Exit**: Press 'q' to quit
6. **Reset**: Press 'r' to reset system state
7. **Debug**: Press 'd' to toggle debug visualization

## 🔧 Configuration

Edit [gesture_controller/config.py](gesture_controller/config.py) to customize:

- **Detection Thresholds**: Confidence, pinch distance, swipe parameters
- **Screen Mapping**: Coordinate scaling and offset
- **Gestures**: Add or modify gesture definitions
- **Interaction Modes**: Define state transitions
- **Display Settings**: FPS, landmarks, gesture labels visibility

### Key Configuration Parameters

```python
# Detection thresholds
min_detection_confidence: 0.7
pinch_threshold: 0.05  # Distance between thumb and index
swipe_distance_threshold: 0.1
swipe_time_threshold: 0.5 seconds

# Motion smoothing
smoothing_alpha: 0.7  # EMA factor (higher = smoother but more lag)

# Debouncing
debounce_delay: 0.2 seconds
```

## 📊 Modules Overview

### 1. **hand_detector.py**
- Detects hand landmarks using MediaPipe
- Extracts 21 keypoints per hand
- Handles multiple hands simultaneously
- Provides visualization utilities

### 2. **gesture_recognizer.py**
- Identifies hand gestures from landmarks
- Supports static gestures (pinch, fist, palm)
- Detects motion gestures (swipes)
- Uses temporal filtering for stability

### 3. **interaction_logic.py**
- Maps gestures to system actions
- Manages interaction modes (IDLE, CONTROL, LOCKED)
- Executes system commands via PyAutoGUI
- Applies cursor smoothing and debouncing

### 4. **config.py**
- Centralized configuration management
- Gesture type enumeration
- Interaction mode definitions
- Gesture-to-action mappings

### 5. **utils.py**
- Point class for coordinate handling
- Exponential Moving Average filter for smoothing
- Motion tracker for swipe detection
- Debouncer for repeated gesture prevention
- Mathematical utilities (distance, angle calculations)

## 🎯 Core Algorithms

### Hand Landmark Detection
- 21-point skeletal model using MediaPipe's pre-trained model
- Real-time detection at webcam resolution
- Handles multiple hands and different hand poses

### Gesture Recognition
1. **Static Gesture Detection**:
   - Finger state analysis (up/down based on joint positions)
   - Distance measurements (e.g., thumb-index for pinch)
   - Finger configuration patterns

2. **Motion Gesture Detection**:
   - Position history tracking
   - Displacement calculation
   - Direction determination (left/right/up/down)
   - Velocity computation

### Smoothing Algorithm
- **Exponential Moving Average (EMA)**:
  - Reduces jitter in cursor movement
  - Configurable smoothing factor (0-1)
  - Independent X, Y, Z filtering

### Debouncing Mechanism
- Prevents rapid repeated gestures
- Configurable time window
- Per-gesture tracking

## 🧪 Testing & Debugging

### Debug Mode
Enable debug visualization with the `-d` flag:
```bash
python main.py --debug
```

This displays:
- FPS counter
- Hand landmarks with connections
- Landmark indices
- Current gesture and mode
- Cursor position
- Confidence scores

### Console Output
Monitor gesture detection and actions:
```
Hand 0: Gesture=index_finger_up, Mode=control, Action=cursor_move
Hand 0: Gesture=pinch, Mode=control, Action=click
```

### Performance Monitoring
- Track FPS in real-time
- Monitor gesture recognition accuracy
- Analyze false positive rates
- Profile interaction latency

## 📈 Advanced Features (Roadmap)

- [ ] **Multi-Hand Interaction:** One hand for navigation, another for commands
- [ ] **Context-Aware Control:** Adapt gestures based on active application
- [ ] **Machine Learning Classifier:** Train custom models for improved accuracy
- [ ] **Virtual Interaction Zones:** Define regions for special actions
- [ ] **Gesture Recording:** Record and replay gesture sequences
- [ ] **Configuration UI:** GUI-based settings management
- [ ] **Hand Pose Classification:** Recognize complex hand poses
- [ ] **Gesture Macros:** Create custom multi-step gesture sequences

## ⚙️ System Requirements

### Hardware
- Modern webcam (720p+ recommended)
- CPU: Intel i5/Ryzen 5 or better
- RAM: 4GB minimum, 8GB recommended
- GPU: Optional (CUDA for faster processing)

### Software
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- Pillow

## 🐛 Troubleshooting

### Camera not found
```bash
python main.py --camera 1  # Try different camera ID
```

### Low FPS / Slow performance
- Reduce input resolution in `main.py`
- Disable landmark visualization
- Use GPU acceleration (if available)
- Close other resource-intensive applications

### Gestures not recognized
- Adjust detection thresholds in `config.py`
- Ensure adequate lighting
- Position hand closer to camera
- Check MediaPipe confidence score

### PyAutoGUI errors
- Ensure mouse/keyboard control is allowed
- On macOS: Grant accessibility permissions
- Disable mouse acceleration for smoother movement

## 📚 References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyAutoGUI Documentation](https://pyautogui.readthedocs.io/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional gesture types
- [ ] Performance optimization
- [ ] Multi-platform testing
- [ ] ML-based gesture recognition
- [ ] Unit tests and integration tests
- [ ] Documentation improvements
- [ ] Configuration GUI

## 📄 License

This project is open source and available for educational and research purposes.

## 🎓 Learning Outcomes

This project demonstrates:

- **Computer Vision**: Hand detection and landmark extraction
- **Real-time Processing**: Low-latency gesture recognition
- **State Machines**: Interaction mode management
- **Signal Processing**: Filtering and smoothing techniques
- **Human-Computer Interaction**: Gesture-based interfaces
- **Robotics Principles**: Sensing, decision-making, and actuation
- **Software Engineering**: Modular design, configuration management, error handling

## 🙏 Acknowledgments

- **MediaPipe**: For the robust hand detection model
- **OpenCV**: For image processing capabilities
- **PyAutoGUI**: For system command execution

## 📞 Support

For issues, questions, or suggestions, please create an issue or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Status**: Active Development
