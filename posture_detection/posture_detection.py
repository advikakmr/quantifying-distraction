import cv2
import math
import mediapipe as mp
from posture_detection.utils import init_pose_landmarker, extract_landmarks
import threading

latest_landmarks = None
latest_prob = None
lock = threading.Lock()
timestamp_ms = 0

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# determined from collect_data.py
MAX_SHOULDER_ANGLE = 15.0   # shoulder tilt from horizontal
MAX_HEAD_ANGLE = 35.0   # ear tilt from horizontal
MAX_NECK_ANGLE = 15.0   # nose lean from vertical

# weights
W_SHOULDER = 0.34
W_HEAD = 0.33
W_NECK = 0.33

MIN_VISIBILITY = 0.5
BAD_POSTURE_THRESHOLD = 0.50  # prob above this → "Bad Posture"

def _angle_from_horizontal(x1, y1, x2, y2) -> float:
    return abs(math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))))

def bad_posture_probability(landmarks) -> float:
    groups = [landmarks[i * 4:(i + 1) * 4] for i in range(17)]  # [x, y, z, vis]

    nose = groups[NOSE]
    l_ear = groups[LEFT_EAR]
    r_ear = groups[RIGHT_EAR]
    l_sh = groups[LEFT_SHOULDER]
    r_sh = groups[RIGHT_SHOULDER]

    score = 0.0
    total_weight = 0.0

    if l_sh[3] > MIN_VISIBILITY and r_sh[3] > MIN_VISIBILITY:
        theta = _angle_from_horizontal(l_sh[0], l_sh[1], r_sh[0], r_sh[1])
        score += W_SHOULDER * min(theta / MAX_SHOULDER_ANGLE, 1.0)
        total_weight += W_SHOULDER

    if l_ear[3] > MIN_VISIBILITY and r_ear[3] > MIN_VISIBILITY:
        theta = _angle_from_horizontal(l_ear[0], l_ear[1], r_ear[0], r_ear[1])
        score += W_HEAD * min(theta / MAX_HEAD_ANGLE, 1.0)
        total_weight += W_HEAD

    if nose[3] > MIN_VISIBILITY and l_sh[3] > MIN_VISIBILITY and r_sh[3] > MIN_VISIBILITY:
        mid_x = (l_sh[0] + r_sh[0]) / 2
        mid_y = (l_sh[1] + r_sh[1]) / 2
        vert = mid_y - nose[1] 
        horiz = abs(nose[0] - mid_x)
        if vert > 0:
            theta = math.degrees(math.atan2(horiz, vert))
            score += W_NECK * min(theta / MAX_NECK_ANGLE, 1.0)
            total_weight += W_NECK

    if total_weight == 0.0:
        return 0.0

    return score / total_weight

def main_callback(result, _image, _timestamp_ms):
    global latest_landmarks, latest_prob
    landmarks = extract_landmarks(result)
    if landmarks is not None:
        prob = bad_posture_probability(landmarks)
        with lock:
            latest_landmarks = landmarks
            latest_prob = prob

pose_landmarker = init_pose_landmarker(callback=main_callback)

def detect_posture(frame):
    global timestamp_ms
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 1
    pose_landmarker.detect_async(mp_image, timestamp_ms)

def draw_posture_detection(frame):
    global latest_prob, latest_landmarks

    color = "#009138"
    pct = 100

    with lock:
        if latest_landmarks:
            h, w, _ = frame.shape
            groups = [latest_landmarks[i * 4:(i + 1) * 4] for i in range(17)]

            for lm in groups:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
            for a, b in connections:
                x1, y1 = int(groups[a][0] * w), int(groups[a][1] * h)
                x2, y2 = int(groups[b][0] * w), int(groups[b][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if latest_prob is not None:
                pct = 100 - int(latest_prob * 100)
                color = "#ad3a00" if latest_prob >= BAD_POSTURE_THRESHOLD else "#009138"

    return frame, color, pct