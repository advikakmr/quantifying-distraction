import cv2
import mediapipe as mp
import threading
import time
import collections
import statistics
from eye_detection.utils import (
    init_face_landmarker, eye_aspect_ratio, gaze_ratio,
    LEFT_EYE, RIGHT_EYE,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_INNER, LEFT_EYE_OUTER,
    RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
)

# determined from collect_data.py
EAR_THRESHOLD = 0.18   # below = eyes closed
GAZE_THRESHOLD = 0.45   # below or above (1 - x) = looking sideways

WINDOW_SECONDS = 5.0    # rolling window length for focus score
STABILITY_NORM = 0.15   # gaze std dev that maps to 0 stability (fully unstable)
FOCUS_THRESHOLD = 0.5    # score below this → "Distracted"

lock = threading.Lock()
latest_landmarks = None
timestamp_ms = 0

# Each entry: (timestamp, is_on_screen: bool, gaze_value: float)
_gaze_history: collections.deque = collections.deque()


def _compute_focus_score(now: float, is_on_screen: bool, gaze_value: float) -> float:
    """
    Append the current frame and return:
        0.7 * (on-screen frames / total frames) + 0.3 * gaze_stability
    over the last WINDOW_SECONDS.
    """
    _gaze_history.append((now, is_on_screen, gaze_value))
    while _gaze_history and (now - _gaze_history[0][0]) > WINDOW_SECONDS:
        _gaze_history.popleft()

    total = len(_gaze_history)
    if total == 0:
        return 0.0

    on_screen_ratio = sum(1 for _, on, _ in _gaze_history if on) / total

    # Stability: std dev of gaze values during on-screen frames only
    gaze_vals = [g for _, on, g in _gaze_history if on]
    if len(gaze_vals) >= 2:
        stability = max(0.0, 1.0 - statistics.stdev(gaze_vals) / STABILITY_NORM)
    else:
        stability = 1.0 if gaze_vals else 0.0

    return 0.7 * on_screen_ratio + 0.3 * stability


def _callback(result, _image, _ts):
    global latest_landmarks
    with lock:
        latest_landmarks = result.face_landmarks[0] if result.face_landmarks else None


face_landmarker = init_face_landmarker(callback=_callback)


def detect_eyes(frame):
    global timestamp_ms
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 1
    face_landmarker.detect_async(mp_image, timestamp_ms)


def draw_eye_detection(frame):
    h, w = frame.shape[:2]

    with lock:
        lms = latest_landmarks

    if lms is None:
        score = _compute_focus_score(time.time(), False, 0.5)
        pct = int(score * 100)
        color = "#009138" if score >= FOCUS_THRESHOLD else "#ad3a00"
        return frame, color, pct

    left_ear = eye_aspect_ratio(lms, LEFT_EYE, w, h)
    right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
    avg_ear = (left_ear + right_ear) / 2.0

    left_gaze = gaze_ratio(lms, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER)
    right_gaze = gaze_ratio(lms, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
    avg_gaze = (left_gaze + right_gaze) / 2.0

    eyes_closed = avg_ear < EAR_THRESHOLD
    looking_away = avg_gaze < GAZE_THRESHOLD or avg_gaze > (1 - GAZE_THRESHOLD)
    is_on_screen = not eyes_closed and not looking_away

    score = _compute_focus_score(time.time(), is_on_screen, avg_gaze)
    pct = int(score * 100)

    color = "#009138" if score >= FOCUS_THRESHOLD else "#ad3a00"

    for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
        cx, cy = int(lms[idx].x * w), int(lms[idx].y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)

    return frame, color, pct
