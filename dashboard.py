"""Streamlit dashboard for gesture control.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import time
import threading
import importlib
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

try:
    st_autorefresh = importlib.import_module("streamlit_autorefresh").st_autorefresh
except Exception:
    def st_autorefresh(*args, **kwargs):
        return None

from gesture_controller import HandDetector, GestureRecognizer
from gesture_controller.config import (
    DEFAULT_SCREEN_MAPPING,
    DEFAULT_SYSTEM_CONFIG,
    DEFAULT_THRESHOLDS,
    GESTURE_ACTIONS,
    GestureType,
)
from gesture_controller.interaction_logic import InteractionLogic


ACTION_DESCRIPTIONS = {
    "cursor_move": "Move mouse cursor with index finger.",
    "click": "Single click when pinch is released.",
    "drag": "Pinch and move to click-and-drag.",
    "scroll": "Open palm vertical motion scrolls page.",
    "toggle_playpause": "Toggle media play or pause.",
    "switch_app": "Switch between applications.",
    "zoom_in": "Zoom in on current app content.",
    "zoom_out": "Zoom out on current app content.",
    "previous_tab": "Move to previous browser or editor tab.",
    "next_tab": "Move to next browser or editor tab.",
    "volume_up": "Increase system output volume.",
    "volume_down": "Decrease system output volume.",
    "reset": "Reset or neutral action state.",
    "fist_motion": "Lock mode trigger action family.",
    "none": "No direct system action.",
}


def resolve_logo_path() -> Optional[Path]:
    """Find logo file in assets folder using preferred names, then fallback to first image."""
    assets_dir = Path(__file__).resolve().parent / "assets"
    if not assets_dir.exists():
        return None

    preferred_names = [
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
    ]
    for name in preferred_names:
        candidate = assets_dir / name
        if candidate.exists():
            return candidate

    image_candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_candidates.extend(sorted(assets_dir.glob(pattern)))

    return image_candidates[0] if image_candidates else None


@dataclass
class RuntimeState:
    running: bool = False
    interaction_enabled: bool = True
    camera_id: int = 0
    started_at: Optional[float] = None
    frames_processed: int = 0
    fps: float = 0.0
    current_gesture: str = GestureType.UNKNOWN.value
    current_action: str = "none"
    hand_detected: bool = False
    model_assist_used: bool = False
    model_assist_gesture: str = GestureType.UNKNOWN.value
    model_assist_confidence: float = 0.0
    model_class_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_error: str = ""
    latest_frame_rgb: Optional[np.ndarray] = None


class DetectionRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = RuntimeState()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._interaction_enabled = True

    def start(self, camera_id: int, interaction_enabled: bool) -> Dict[str, Any]:
        with self._lock:
            if self._state.running:
                return {"ok": False, "message": "Detection already running."}

            self._state = RuntimeState(
                running=True,
                interaction_enabled=interaction_enabled,
                camera_id=camera_id,
                started_at=time.time(),
            )
            self._interaction_enabled = interaction_enabled
            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run_loop,
                args=(camera_id,),
                name="gesture-detection-runtime",
                daemon=True,
            )
            self._thread.start()

        return {"ok": True, "message": "Detection started."}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._state.running:
                return {"ok": False, "message": "Detection is not running."}

            self._stop_event.set()
            runtime_thread = self._thread

        if runtime_thread is not None:
            runtime_thread.join(timeout=3.0)

        with self._lock:
            self._state.running = False

        return {"ok": True, "message": "Detection stopped."}

    def set_interaction_enabled(self, enabled: bool) -> Dict[str, Any]:
        with self._lock:
            self._interaction_enabled = enabled
            self._state.interaction_enabled = enabled
        return {"ok": True, "message": "Interaction mode updated."}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "running": self._state.running,
                "interaction_enabled": self._state.interaction_enabled,
                "camera_id": self._state.camera_id,
                "uptime_seconds": (
                    round(time.time() - self._state.started_at, 1)
                    if self._state.started_at is not None
                    else 0.0
                ),
                "frames_processed": self._state.frames_processed,
                "fps": round(self._state.fps, 2),
                "current_gesture": self._state.current_gesture,
                "current_action": self._state.current_action,
                "hand_detected": self._state.hand_detected,
                "model_assist_used": self._state.model_assist_used,
                "model_assist_gesture": self._state.model_assist_gesture,
                "model_assist_confidence": round(self._state.model_assist_confidence, 3),
                "model_class_stats": self._state.model_class_stats,
                "last_error": self._state.last_error,
            }
        return payload

    def frame_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._state.latest_frame_rgb is None else self._state.latest_frame_rgb.copy()

    def _run_loop(self, camera_id: int):
        cap = None
        detector = None
        recognizer = None
        logic = None

        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            cap.set(cv2.CAP_PROP_FPS, 60)

            detector = HandDetector(DEFAULT_THRESHOLDS)
            recognizer = GestureRecognizer(DEFAULT_THRESHOLDS)
            logic = InteractionLogic(
                screen_width=DEFAULT_SCREEN_MAPPING.screen_width,
                screen_height=DEFAULT_SCREEN_MAPPING.screen_height,
                thresholds=DEFAULT_THRESHOLDS,
                target_fps=DEFAULT_SYSTEM_CONFIG.target_fps,
                enable_actions=self._interaction_enabled,
            )

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._set_error("Failed to read frame from camera.")
                    break

                if DEFAULT_SYSTEM_CONFIG.flip_frame:
                    frame = cv2.flip(frame, 1)

                frame_time = time.time()
                hands, _ = detector.detect(frame)

                with self._lock:
                    logic.enable_actions = self._interaction_enabled

                hand_detected = False
                current_gesture = GestureType.UNKNOWN.value
                current_action = "none"
                model_assist_used = False
                model_assist_gesture = GestureType.UNKNOWN.value
                model_assist_confidence = 0.0
                model_class_stats: Dict[str, Dict[str, Any]] = {}

                if hands:
                    hand_detected = True
                    hand = hands[0]
                    gesture, details = recognizer.recognize(hand, 0, frame_time)
                    status = logic.update(gesture, hand, frame_time)

                    current_gesture = gesture.value
                    current_action = status.get("action", "none")
                    model_assist_used = bool(details.get("model_assist_used", False))
                    model_assist_gesture = str(details.get("model_assist_gesture", GestureType.UNKNOWN.value))
                    model_assist_confidence = float(details.get("model_assist_confidence", 0.0))
                    model_class_stats = details.get("model_class_stats", {}) or {}

                    cv2.putText(frame, f"Gesture: {current_gesture}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, f"Action: {current_action}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                    cv2.putText(frame, f"Model: {model_assist_gesture} ({model_assist_confidence:.2f})", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._state.latest_frame_rgb = display_frame

                with self._lock:
                    self._state.frames_processed += 1
                    self._state.fps = logic.get_fps()
                    self._state.current_gesture = current_gesture
                    self._state.current_action = current_action
                    self._state.hand_detected = hand_detected
                    self._state.model_assist_used = model_assist_used
                    self._state.model_assist_gesture = model_assist_gesture
                    self._state.model_assist_confidence = model_assist_confidence
                    self._state.model_class_stats = model_class_stats
                    self._state.last_error = ""

            with self._lock:
                self._state.running = False
                self._state.latest_frame_rgb = None

        except Exception as exc:
            self._set_error(str(exc))
            with self._lock:
                self._state.running = False
                self._state.latest_frame_rgb = None

        finally:
            if logic is not None:
                logic.close()
            if detector is not None:
                detector.close()
            if cap is not None:
                cap.release()

    def _set_error(self, error: str):
        with self._lock:
            self._state.last_error = error


@st.cache_resource
def get_runtime() -> DetectionRuntime:
    return DetectionRuntime()


def render_header():
    st.set_page_config(page_title="Gesture Dashboard", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background: #ffffff !important;
                color: #111111 !important;
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stSidebar"] {
                background: #ffffff !important;
            }
            [data-testid="stMarkdownContainer"],
            [data-testid="stMetricValue"],
            [data-testid="stMetricLabel"],
            .stCaption,
            .stText,
            p, span, label, h1, h2, h3, h4 {
                color: #111111 !important;
            }
            .block-container {padding-top: 1.0rem; max-width: 1300px;}
            .guide-box {
                padding: 12px 14px;
                border: 1px solid #d8dde3;
                border-radius: 12px;
                background: #ffffff;
                color: #111111;
            }
            .hero {
                border: 1px solid #e3e8ee;
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                border-radius: 16px;
                padding: 16px 18px;
                box-shadow: 0 10px 26px rgba(16, 24, 40, 0.06);
                margin-bottom: 10px;
            }
            .hero-logo-wrap {
                margin-top: 18px;
                padding-left: 8px;
                padding-right: 8px;
            }
            .hero-title {
                margin: 0;
                font-size: 1.45rem;
                font-weight: 700;
                letter-spacing: -0.01em;
                color: #0e2233;
            }
            .hero-sub {
                margin: 6px 0 0 0;
                color: #385063;
                font-size: 0.95rem;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
                margin: 8px 0 4px 0;
            }
            .status-card {
                border: 1px solid #e5eaf0;
                border-radius: 12px;
                padding: 10px 12px;
                background: #ffffff;
                box-shadow: 0 4px 14px rgba(16, 24, 40, 0.05);
            }
            .status-label {
                font-size: 0.72rem;
                color: #5a7186;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 4px;
            }
            .status-value {
                font-size: 1.02rem;
                font-weight: 700;
                color: #132a3a;
                line-height: 1.15;
            }
            .status-running { color: #0e9f6e; }
            .status-idle { color: #c07000; }
            .tiny-muted {
                font-size: .82rem;
                color: #4f6470 !important;
            }
            .stButton > button {
                border-radius: 10px !important;
                border: 1px solid #cfd9e3 !important;
                padding-top: 0.48rem !important;
                padding-bottom: 0.48rem !important;
            }
            .stButton > button[kind="primary"] {
                background: #1763d1 !important;
                color: #ffffff !important;
                border-color: #1763d1 !important;
            }
            .stButton > button:hover {
                border-color: #8ea8c0 !important;
            }
            [data-baseweb="tab-list"] button {
                border-radius: 10px 10px 0 0 !important;
                font-weight: 600 !important;
            }
            [data-baseweb="tab-panel"] {
                padding-top: 10px;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                overflow: hidden;
            }
            .tiny {font-size: .85rem; color: #2f4652 !important;}
            @media (max-width: 960px) {
                .status-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_path = resolve_logo_path()

    hero_left, hero_right = st.columns([0.2, 0.8], gap="medium")
    with hero_left:
        st.markdown("<div class='hero-logo-wrap'>", unsafe_allow_html=True)
        if logo_path is not None:
            st.image(str(logo_path), width=260)
        else:
            st.markdown("<div class='hero' style='text-align:center;font-size:2rem;'>AI</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with hero_right:
        st.markdown(
            """
            <div class="hero">
                <h1 class="hero-title">Intelligent Gesture Based Human Computer Interaction System</h1>
                <p class="hero-sub">Production-style control center for detection, interaction, camera preview, and model readiness diagnostics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_controls(runtime: DetectionRuntime):
    st.subheader("Runtime Control")
    camera_id = st.number_input("Camera ID", min_value=0, value=0, step=1)
    interaction_enabled = st.toggle("Interaction Enabled", value=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start Detection", use_container_width=True, type="primary"):
            result = runtime.start(camera_id=int(camera_id), interaction_enabled=interaction_enabled)
            (st.success if result["ok"] else st.error)(result["message"])

    with c2:
        if st.button("Stop Detection", use_container_width=True):
            result = runtime.stop()
            (st.success if result["ok"] else st.warning)(result["message"])

    with c3:
        if st.button("Apply Interaction", use_container_width=True):
            result = runtime.set_interaction_enabled(interaction_enabled)
            (st.success if result["ok"] else st.error)(result["message"])

    st.markdown('<div class="tiny-muted">Refresh is automatic while running. You can still click Rerun any time.</div>', unsafe_allow_html=True)


def render_camera_preview(runtime: DetectionRuntime):
    st.subheader("Camera Preview")
    frame_getter = getattr(runtime, "frame_rgb", None)
    if frame_getter is None:
        frame_getter = getattr(runtime, "frame_bytes", None)

    frame_rgb = frame_getter() if callable(frame_getter) else None

    if frame_rgb is None:
        st.info("Start detection to see the live camera feed here.")
        return

    st.image(frame_rgb, channels="RGB", use_container_width=True)
    st.caption("Live preview updates while detection is running.")


def render_status_cards(status: Dict[str, Any]):
    st.subheader("Live Status")
    runtime_text = "running" if status["running"] else "idle"
    runtime_class = "status-running" if status["running"] else "status-idle"
    interaction_text = "enabled" if status["interaction_enabled"] else "disabled"
    hand_text = "detected" if status["hand_detected"] else "none"

    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card"><div class="status-label">Runtime</div><div class="status-value {runtime_class}">{runtime_text}</div></div>
            <div class="status-card"><div class="status-label">FPS</div><div class="status-value">{status['fps']:.2f}</div></div>
            <div class="status-card"><div class="status-label">Frames</div><div class="status-value">{status['frames_processed']}</div></div>
            <div class="status-card"><div class="status-label">Uptime</div><div class="status-value">{status['uptime_seconds']:.1f}s</div></div>
            <div class="status-card"><div class="status-label">Interaction</div><div class="status-value">{interaction_text}</div></div>
            <div class="status-card"><div class="status-label">Gesture</div><div class="status-value">{status['current_gesture']}</div></div>
            <div class="status-card"><div class="status-label">Action</div><div class="status-value">{status['current_action']}</div></div>
            <div class="status-card"><div class="status-label">Hand</div><div class="status-value">{hand_text}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_details(status: Dict[str, Any]):
    st.subheader("Model and Training Details")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Model Assist**")
        st.write(f"Used: {'yes' if status['model_assist_used'] else 'no'}")
        st.write(f"Guess: {status['model_assist_gesture']}")
        st.write(f"Confidence: {status['model_assist_confidence']:.2f}")

    with b2:
        st.markdown("**Per-Gesture Samples**")
        stats = status.get("model_class_stats", {}) or {}
        if not stats:
            st.info("No sample stats yet.")
        else:
            labels = sorted(stats.keys())
            ready_count = sum(1 for label in labels if bool(stats[label].get("ready", False)))
            st.write(f"Ready classes: {ready_count}/{len(labels)}")
            for label in labels:
                entry = stats[label]
                marker = "OK" if bool(entry.get("ready", False)) else ".."
                count = int(entry.get("count", 0))
                st.code(f"{marker} {label}: {count}", language="text")

    if status.get("last_error"):
        st.error(status["last_error"])


def render_quick_guide():
    st.subheader("Quick Guide")
    st.markdown('<div class="guide-box"><strong>1. Start safe:</strong> start with interaction disabled and verify gestures.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>2. Enable interaction:</strong> switch on once labels are stable.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>3. Emergency stop:</strong> use Stop Detection immediately when needed.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>4. Stability tips:</strong> even lighting, plain background, steady hand depth.</div>', unsafe_allow_html=True)


def render_gesture_reference():
    st.subheader("Gesture and Action Reference")
    st.caption("Complete mapping of supported gestures to app actions.")

    rows = []
    for gesture in sorted(GESTURE_ACTIONS.keys(), key=lambda g: g.value):
        action = GESTURE_ACTIONS[gesture]
        rows.append(
            {
                "Gesture": gesture.value,
                "Action": action,
                "Effect": ACTION_DESCRIPTIONS.get(action, "Action behavior is defined in interaction logic."),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### Practical Control Actions")
    st.write("You can perform cursor movement, click, drag, scroll, tab switching, app switching, volume control, zoom, and media play or pause.")


def main():
    render_header()
    runtime = get_runtime()
    st_autorefresh(interval=300, key="dashboard_refresh")
    tab_dashboard, tab_reference = st.tabs(["Control Center", "Gestures and Actions"])

    with tab_dashboard:
        status = runtime.status()
        render_status_cards(status)
        st.divider()
        left, right = st.columns([1.0, 1.2], gap="large")
        with left:
            render_controls(runtime)
        with right:
            render_camera_preview(runtime)
        st.divider()
        render_status_details(status)

    with tab_reference:
        render_quick_guide()
        st.divider()
        render_gesture_reference()


if __name__ == "__main__":
    main()
