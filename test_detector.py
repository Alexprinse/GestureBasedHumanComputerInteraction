"""Standalone detector calibration utility.

Run:
    python test_detector.py

Keys:
    q: quit
    m: toggle mask preview
    c: calibrate from center ROI
    p: save detector profile
    l: load detector profile
"""

import cv2
import time
import numpy as np

from gesture_controller.config import DEFAULT_THRESHOLDS
from gesture_controller.hand_detector import HandDetector


WINDOW_NAME = "Detector Calibration"
MASK_WINDOW = "Skin Mask"
TUNER_WINDOW = "Detector Tuner"


def _noop(_):
    pass


def _clamp(min_val: int, value: int, max_val: int) -> int:
    return max(min_val, min(value, max_val))


def _create_trackbars(detector: HandDetector):
    cv2.namedWindow(TUNER_WINDOW)

    cv2.createTrackbar("H low", TUNER_WINDOW, int(detector.lower_skin[0]), 179, _noop)
    cv2.createTrackbar("S low", TUNER_WINDOW, int(detector.lower_skin[1]), 255, _noop)
    cv2.createTrackbar("V low", TUNER_WINDOW, int(detector.lower_skin[2]), 255, _noop)
    cv2.createTrackbar("H high", TUNER_WINDOW, int(detector.upper_skin[0]), 179, _noop)
    cv2.createTrackbar("S high", TUNER_WINDOW, int(detector.upper_skin[1]), 255, _noop)
    cv2.createTrackbar("V high", TUNER_WINDOW, int(detector.upper_skin[2]), 255, _noop)

    cv2.createTrackbar("min area x1000", TUNER_WINDOW, int(detector._min_area_ratio * 1000), 400, _noop)
    cv2.createTrackbar("max area x1000", TUNER_WINDOW, int(detector._max_area_ratio * 1000), 900, _noop)
    cv2.createTrackbar("min solidity x100", TUNER_WINDOW, int(detector._min_solidity * 100), 100, _noop)
    cv2.createTrackbar("max aspect x100", TUNER_WINDOW, int(detector._max_aspect_ratio * 100), 400, _noop)
    cv2.createTrackbar("min extent x100", TUNER_WINDOW, int(detector._min_extent * 100), 100, _noop)
    cv2.createTrackbar("stable frames", TUNER_WINDOW, int(detector._required_stable_frames), 20, _noop)
    cv2.createTrackbar("face overlap x100", TUNER_WINDOW, int(detector._face_overlap_threshold * 100), 100, _noop)


def _set_trackbars_from_detector(detector: HandDetector):
    cv2.setTrackbarPos("H low", TUNER_WINDOW, int(detector.lower_skin[0]))
    cv2.setTrackbarPos("S low", TUNER_WINDOW, int(detector.lower_skin[1]))
    cv2.setTrackbarPos("V low", TUNER_WINDOW, int(detector.lower_skin[2]))
    cv2.setTrackbarPos("H high", TUNER_WINDOW, int(detector.upper_skin[0]))
    cv2.setTrackbarPos("S high", TUNER_WINDOW, int(detector.upper_skin[1]))
    cv2.setTrackbarPos("V high", TUNER_WINDOW, int(detector.upper_skin[2]))
    cv2.setTrackbarPos("min area x1000", TUNER_WINDOW, int(detector._min_area_ratio * 1000))
    cv2.setTrackbarPos("max area x1000", TUNER_WINDOW, int(detector._max_area_ratio * 1000))
    cv2.setTrackbarPos("min solidity x100", TUNER_WINDOW, int(detector._min_solidity * 100))
    cv2.setTrackbarPos("max aspect x100", TUNER_WINDOW, int(detector._max_aspect_ratio * 100))
    cv2.setTrackbarPos("min extent x100", TUNER_WINDOW, int(detector._min_extent * 100))
    cv2.setTrackbarPos("stable frames", TUNER_WINDOW, int(detector._required_stable_frames))
    cv2.setTrackbarPos("face overlap x100", TUNER_WINDOW, int(detector._face_overlap_threshold * 100))


def _sync_detector_from_trackbars(detector: HandDetector):
    h_low = cv2.getTrackbarPos("H low", TUNER_WINDOW)
    s_low = cv2.getTrackbarPos("S low", TUNER_WINDOW)
    v_low = cv2.getTrackbarPos("V low", TUNER_WINDOW)
    h_high = cv2.getTrackbarPos("H high", TUNER_WINDOW)
    s_high = cv2.getTrackbarPos("S high", TUNER_WINDOW)
    v_high = cv2.getTrackbarPos("V high", TUNER_WINDOW)

    h_high = max(h_high, h_low + 1)
    s_high = max(s_high, s_low + 1)
    v_high = max(v_high, v_low + 1)

    detector.lower_skin[0] = _clamp(0, h_low, 179)
    detector.lower_skin[1] = _clamp(0, s_low, 255)
    detector.lower_skin[2] = _clamp(0, v_low, 255)
    detector.upper_skin[0] = _clamp(0, h_high, 179)
    detector.upper_skin[1] = _clamp(0, s_high, 255)
    detector.upper_skin[2] = _clamp(0, v_high, 255)

    detector._min_area_ratio = max(0.001, cv2.getTrackbarPos("min area x1000", TUNER_WINDOW) / 1000.0)
    detector._max_area_ratio = max(
        detector._min_area_ratio + 0.001,
        cv2.getTrackbarPos("max area x1000", TUNER_WINDOW) / 1000.0,
    )
    detector._min_solidity = cv2.getTrackbarPos("min solidity x100", TUNER_WINDOW) / 100.0
    detector._max_aspect_ratio = max(0.35, cv2.getTrackbarPos("max aspect x100", TUNER_WINDOW) / 100.0)
    detector._min_extent = cv2.getTrackbarPos("min extent x100", TUNER_WINDOW) / 100.0
    detector._required_stable_frames = max(0, cv2.getTrackbarPos("stable frames", TUNER_WINDOW))
    detector._face_overlap_threshold = cv2.getTrackbarPos("face overlap x100", TUNER_WINDOW) / 100.0


def _draw_tuner_panel(detector: HandDetector):
    panel = np.zeros((260, 620, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    lines = [
        "Tune sliders in this window (Detector Tuner)",
        f"HSV: H[{int(detector.lower_skin[0])},{int(detector.upper_skin[0])}] "
        f"S[{int(detector.lower_skin[1])},{int(detector.upper_skin[1])}] "
        f"V[{int(detector.lower_skin[2])},{int(detector.upper_skin[2])}]",
        f"Area ratio: min={detector._min_area_ratio:.3f} max={detector._max_area_ratio:.3f}",
        f"Solidity min={detector._min_solidity:.2f} extent min={detector._min_extent:.2f}",
        f"Aspect max={detector._max_aspect_ratio:.2f} stable frames={detector._required_stable_frames}",
        f"Face overlap reject>{detector._face_overlap_threshold:.2f}",
        "Recommended first tweaks: lower min area, lower stable frames,",
        "then widen H high / lower V low if hand is not detected.",
    ]

    y = 30
    for i, line in enumerate(lines):
        color = (0, 220, 255) if i == 0 else (220, 220, 220)
        cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        y += 28

    cv2.imshow(TUNER_WINDOW, panel)


def main():
    detector = HandDetector(DEFAULT_THRESHOLDS)
    _create_trackbars(detector)
    _set_trackbars_from_detector(detector)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    show_mask = False

    total_frames = 0
    frames_with_hand = 0
    start = time.time()

    print("Detector calibration started")
    print("Controls: q=quit, m=toggle mask, c=calibrate, p=save, l=load")
    print("Use Detector Tuner trackbars to adjust live thresholds")
    print(f"Profile path: {detector.get_profile_path()}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read camera frame")
            break

        frame = cv2.flip(frame, 1)

        fh, fw = frame.shape[:2]
        roi_w, roi_h = int(fw * 0.22), int(fh * 0.45)
        roi_x = (fw - roi_w) // 2
        roi_y = (fh - roi_h) // 2

        _sync_detector_from_trackbars(detector)
        _draw_tuner_panel(detector)

        hands, annotated = detector.detect(frame)
        annotated = detector.draw_landmarks(annotated, hands, draw_connections=True, draw_labels=False)

        cv2.rectangle(annotated, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "Place palm here + press C",
            (roi_x - 6, max(20, roi_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        total_frames += 1
        if hands:
            frames_with_hand += 1

        elapsed = max(time.time() - start, 1e-6)
        fps = total_frames / elapsed
        detection_rate = (frames_with_hand / total_frames) * 100.0

        status_color = (0, 200, 0) if hands else (0, 120, 255)
        cv2.putText(annotated, f"hands={len(hands)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(annotated, f"fps={fps:.1f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            annotated,
            f"detection_rate={detection_rate:.1f}%",
            (12, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"H[{int(detector.lower_skin[0])},{int(detector.upper_skin[0])}] "
            f"S[{int(detector.lower_skin[1])},{int(detector.upper_skin[1])}] "
            f"V[{int(detector.lower_skin[2])},{int(detector.upper_skin[2])}]",
            (12, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_NAME, annotated)

        if show_mask:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, detector.lower_skin, detector.upper_skin)
            cv2.imshow(MASK_WINDOW, mask)
        else:
            cv2.destroyWindow(MASK_WINDOW)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord("c"):
            if detector.calibrate_from_roi(frame, (roi_x, roi_y, roi_w, roi_h)):
                _set_trackbars_from_detector(detector)
                print("Auto-calibration applied from center ROI")
                print(f"HSV: {detector.lower_skin.tolist()} -> {detector.upper_skin.tolist()}")
                if detector.save_profile():
                    print(f"Profile auto-saved: {detector.get_profile_path()}")
                else:
                    print("Profile auto-save failed")
            else:
                print("Calibration failed, keep palm fully inside ROI and try again")
        if key == ord("p"):
            if detector.save_profile():
                print(f"Profile saved: {detector.get_profile_path()}")
            else:
                print("Profile save failed")
        if key == ord("l"):
            if detector.load_profile():
                _set_trackbars_from_detector(detector)
                print(f"Profile loaded: {detector.get_profile_path()}")
            else:
                print(f"No saved profile found at: {detector.get_profile_path()}")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    elapsed = max(time.time() - start, 1e-6)
    print("=" * 40)
    print(f"frames={total_frames}")
    print(f"duration={elapsed:.1f}s")
    print(f"avg_fps={total_frames / elapsed:.1f}")
    print(f"detection_rate={(frames_with_hand / max(total_frames, 1)) * 100.0:.1f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
