import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Key landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # p1-p6 for EAR
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_INNER, LEFT_EYE_OUTER = 133, 33
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263

model_path = 'eye_tracking/face_landmarker.task'


def init_face_landmarker(callback, model_path=model_path):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        result_callback=callback
    )
    return FaceLandmarker.create_from_options(options)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    import numpy as np
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0


def gaze_ratio(landmarks, iris_idx, inner_idx, outer_idx):
    iris_x = landmarks[iris_idx].x
    inner_x = landmarks[inner_idx].x
    outer_x = landmarks[outer_idx].x
    span = abs(outer_x - inner_x)
    if span < 1e-6:
        return 0.5
    return (iris_x - min(inner_x, outer_x)) / span
