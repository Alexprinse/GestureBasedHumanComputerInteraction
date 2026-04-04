"""Streamlit dashboard for gesture control.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import time
import threading
import importlib
import cv2
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
    GestureType,
)
from gesture_controller.interaction_logic import InteractionLogic


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
            .block-container {padding-top: 1.2rem;}
            .guide-box {padding: 12px 14px; border: 1px solid rgba(40,60,70,.2); border-radius: 12px; background: rgba(248,252,255,.8);}
            .tiny {font-size: .85rem; color: #4f6470;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Gesture Control Dashboard")
    st.caption("Guide + runtime control panel for starting/stopping detection and interaction safely.")


def render_controls(runtime: DetectionRuntime):
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
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

        st.markdown('<div class="tiny">Refresh the page or click Rerun to update status instantly.</div>', unsafe_allow_html=True)

    with right:
        st.subheader("Quick Guide")
        st.markdown('<div class="guide-box"><strong>1. Start safe:</strong> start with interaction disabled and verify gestures.</div>', unsafe_allow_html=True)
        st.markdown('<div class="guide-box"><strong>2. Enable interaction:</strong> switch on once labels are stable.</div>', unsafe_allow_html=True)
        st.markdown('<div class="guide-box"><strong>3. Emergency stop:</strong> use Stop Detection immediately when needed.</div>', unsafe_allow_html=True)
        st.markdown('<div class="guide-box"><strong>4. Stability tips:</strong> even lighting, plain background, steady hand depth.</div>', unsafe_allow_html=True)


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


def render_status(runtime: DetectionRuntime):
    status = runtime.status()

    st.subheader("Live Status")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runtime", "running" if status["running"] else "idle")
    m2.metric("FPS", f"{status['fps']:.2f}")
    m3.metric("Frames", f"{status['frames_processed']}")
    m4.metric("Uptime", f"{status['uptime_seconds']:.1f}s")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Interaction", "enabled" if status["interaction_enabled"] else "disabled")
    a2.metric("Gesture", status["current_gesture"])
    a3.metric("Action", status["current_action"])
    a4.metric("Hand", "detected" if status["hand_detected"] else "none")

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


def main():
    render_header()
    runtime = get_runtime()
    st_autorefresh(interval=300, key="dashboard_refresh")
    render_controls(runtime)
    st.divider()
    render_camera_preview(runtime)
    st.divider()
    render_status(runtime)


if __name__ == "__main__":
    main()
