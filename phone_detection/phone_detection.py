import cv2
from collections import deque
from ultralytics import YOLO
from config import *

MODEL_PATH = "phone_detection/yolo11s.pt"
TARGET_CLASSES = {"remote", "cell phone"}
CONF_THRESHOLD = 0.4
MIN_BOX_AREA_RATIO = 0.01

# Require phone detected in at least SMOOTHING_MIN of the last SMOOTHING_FRAMES
# frames before reporting a detection (eliminates single-frame false positives).
SMOOTHING_FRAMES = 5
SMOOTHING_MIN_DETECTIONS = 3

model = YOLO(MODEL_PATH)
wanted_ids = {i for i, n in model.names.items() if n in TARGET_CLASSES}

latest_detected = 0     # int: 100 = phone detected, 0 = not detected
latest_annotated = None  # YOLO-annotated frame
_detection_buffer: deque = deque(maxlen=SMOOTHING_FRAMES)


def _phone_present(results, frame_area: float) -> bool:
    if not results.boxes or len(results.boxes) == 0:
        return False
    for cls_id, conf, xyxy in zip(
        results.boxes.cls.tolist(),
        results.boxes.conf.tolist(),
        results.boxes.xyxy.tolist(),
    ):
        if cls_id in wanted_ids and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = xyxy
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if box_area / frame_area >= MIN_BOX_AREA_RATIO:
                return True
    return False


def detect_phone(frame):
    global latest_detected, latest_annotated
    H, W = frame.shape[:2]
    results = model(frame, verbose=False)[0]

    annotated = frame.copy()
    if results.boxes and len(results.boxes) > 0:
        for cls_id, conf, xyxy in zip(
            results.boxes.cls.tolist(),
            results.boxes.conf.tolist(),
            results.boxes.xyxy.tolist(),
        ):
            if cls_id in wanted_ids and conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{model.names[int(cls_id)]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    latest_annotated = annotated
    _detection_buffer.append(1 if _phone_present(results, float(H * W)) else 0)
    latest_detected = 1 if sum(_detection_buffer) >= SMOOTHING_MIN_DETECTIONS else 0


def draw_phone_detection(frame):
    out = latest_annotated if latest_annotated is not None else frame
    if latest_detected:
        return out, "#ad3a00", 100
    return out, "#009138", 0
