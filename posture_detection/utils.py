import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = '/Users/AdvikaKumar/Developer/focus_agent/v2/posture_detection/pose_landmarker_full.task'

NUM_LANDMARKS = 17


def init_pose_landmarker(callback, model_path=model_path):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=callback
    )
    return PoseLandmarker.create_from_options(options)


def extract_landmarks(result):
    if not result.pose_landmarks:
        return None

    row = []
    lm_stop_cntr = 0
    for lm in result.pose_landmarks[0]:
        if lm_stop_cntr < NUM_LANDMARKS:
            row += [lm.x, lm.y, lm.z, lm.visibility]
            lm_stop_cntr += 1

    return row