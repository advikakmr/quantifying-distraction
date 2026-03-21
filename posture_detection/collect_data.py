import csv
import math
import os
import statistics
import time

import cv2
import mediapipe as mp

_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_DIR, "pose_landmarker_full.task")
OUTPUT_CSV = os.path.join(_DIR, "posture_data.csv")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
    )
)

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NUM_LANDMARKS = 17
MIN_VIS = 0.5

SKELETON_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]

def _extract(result):
    if not result.pose_landmarks:
        return None
    row = []
    for i, lm in enumerate(result.pose_landmarks[0]):
        if i >= NUM_LANDMARKS:
            break
        row += [lm.x, lm.y, lm.z, lm.visibility]
    return row

def compute_angles(lms):
    g = [lms[i * 4:(i + 1) * 4] for i in range(NUM_LANDMARKS)]

    def horiz(x1, y1, x2, y2):
        return abs(math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1))))

    sh = None
    if g[LEFT_SHOULDER][3] > MIN_VIS and g[RIGHT_SHOULDER][3] > MIN_VIS:
        sh = horiz(g[LEFT_SHOULDER][0], g[LEFT_SHOULDER][1],
                   g[RIGHT_SHOULDER][0], g[RIGHT_SHOULDER][1])

    hd = None
    if g[LEFT_EAR][3] > MIN_VIS and g[RIGHT_EAR][3] > MIN_VIS:
        hd = horiz(g[LEFT_EAR][0], g[LEFT_EAR][1],
                   g[RIGHT_EAR][0], g[RIGHT_EAR][1])

    nk = None
    if (g[NOSE][3] > MIN_VIS
            and g[LEFT_SHOULDER][3] > MIN_VIS
            and g[RIGHT_SHOULDER][3] > MIN_VIS):
        mid_x = (g[LEFT_SHOULDER][0] + g[RIGHT_SHOULDER][0]) / 2
        mid_y = (g[LEFT_SHOULDER][1] + g[RIGHT_SHOULDER][1]) / 2
        vert = mid_y - g[NOSE][1] 
        horiz_dist = abs(g[NOSE][0] - mid_x)
        if vert > 0:
            nk = math.degrees(math.atan2(horiz_dist, vert))

    return sh, hd, nk

def draw_skeleton(frame, lms):
    h, w = frame.shape[:2]
    g = [lms[i * 4:(i + 1) * 4] for i in range(NUM_LANDMARKS)]
    for lm in g:
        cv2.circle(frame, (int(lm[0] * w), int(lm[1] * h)), 4, (0, 255, 0), -1)
    for a, b in SKELETON_CONNECTIONS:
        cv2.line(frame,
                 (int(g[a][0] * w), int(g[a][1] * h)),
                 (int(g[b][0] * w), int(g[b][1] * h)),
                 (255, 0, 0), 2)


def put(frame, text, y, color=(255, 255, 255), scale=0.55, bold=False):
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                2 if bold else 1, cv2.LINE_AA)

csv_file = open(OUTPUT_CSV, "a", newline="")
writer = csv.writer(csv_file)
if os.path.getsize(OUTPUT_CSV) == 0:
    writer.writerow(["timestamp", "shoulder_deg", "head_deg", "neck_deg", "label"])

def print_summary():
    data = {"good": {"sh": [], "hd": [], "nk": []},
            "bad":  {"sh": [], "hd": [], "nk": []}}
    try:
        with open(OUTPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                lbl = row["label"]
                if lbl not in data:
                    continue
                for key, col in [("sh", "shoulder_deg"), ("hd", "head_deg"), ("nk", "neck_deg")]:
                    try:
                        data[lbl][key].append(float(row[col]))
                    except ValueError:
                        pass
    except FileNotFoundError:
        return

    def stats(vals):
        if not vals:
            return "  n/a"
        s = sorted(vals)
        p95 = s[int(len(s) * 0.95)]
        return (f"  n={len(vals):3d}  "
                f"min={min(vals):5.1f}°  "
                f"mean={statistics.mean(vals):5.1f}°  "
                f"max={max(vals):5.1f}°  "
                f"p95={p95:5.1f}°")

    print("\n" + "=" * 62)
    print("POSTURE DATA SUMMARY")
    print("=" * 62)
    print(f"{'':20s}  {'GOOD':>28s}  {'BAD':>28s}")
    print("-" * 62)
    for label, key in [("shoulder tilt", "sh"), ("head tilt", "hd"), ("neck lean", "nk")]:
        print(f"{label:20s}{stats(data['good'][key])}")
        print(f"{'':20s}{stats(data['bad'][key])}")
        print()

    print("max angle values:")
    suggestions = {}
    for var, key, cur in [("MAX_SHOULDER_ANGLE", "sh", 25.0),
                           ("MAX_HEAD_ANGLE",     "hd", 25.0),
                           ("MAX_NECK_ANGLE",     "nk", 20.0)]:
        g95 = sorted(data["good"][key])[int(len(data["good"][key]) * 0.95)] if data["good"][key] else None
        bm  = statistics.mean(data["bad"][key]) if data["bad"][key] else None
        if g95 is not None and bm is not None:
            suggested = round((g95 + bm) / 2, 1)
        else:
            suggested = cur
        suggestions[var] = suggested
        print(f"  {var} = {suggested}")

    print("\nthresholds for posture_detection.py.")


cap = cv2.VideoCapture(0)
counts = {"good": 0, "bad": 0}
flash = None 

print(f"Output: {OUTPUT_CSV}")
print("G = good posture  |  B = bad posture  |  Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    lms = _extract(result)

    sh, hd, nk = (None, None, None)
    all_visible = False
    if lms:
        draw_skeleton(frame, lms)
        sh, hd, nk = compute_angles(lms)
        all_visible = all(v is not None for v in (sh, hd, nk))

    put(frame, "POSTURE DATA COLLECTOR", 22, (0, 220, 220), scale=0.65, bold=True)

    y = 50
    for label, val, mx in [
        ("Shoulder tilt", sh, 25),
        ("Head tilt    ", hd, 25),
        ("Neck lean    ", nk, 20),
    ]:
        val_str = f"{val:5.1f} deg  (max {mx})" if val is not None else "  --"
        put(frame, f"{label}: {val_str}", y)
        y += 20

    y += 6
    status_col = (120, 255, 120) if all_visible else (80, 80, 255)
    status_txt = "Landmarks OK - ready to label" if all_visible else "Waiting for landmarks..."
    put(frame, status_txt, y, status_col)

    y += 20
    put(frame, f"Good: {counts['good']}   Bad: {counts['bad']}   Total: {sum(counts.values())}", y, (190, 190, 190))

    if flash and time.time() < flash[1]:
        col = (40, 200, 40) if flash[0] == "good" else (40, 40, 220)
        put(frame, f"SAVED: {flash[0].upper()}", frame.shape[0] - 45, col, scale=0.75, bold=True)

    put(frame, "[G] Good  [B] Bad  [Q] Quit", frame.shape[0] - 18, (160, 160, 160))

    cv2.imshow("Posture Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    label = None
    if key == ord('g') and all_visible:
        label = "good"
    elif key == ord('b') and all_visible:
        label = "bad"

    if label:
        writer.writerow([round(time.time(), 3), round(sh, 3), round(hd, 3), round(nk, 3), label])
        csv_file.flush()
        counts[label] += 1
        flash = (label, time.time() + 1.2)

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print_summary()
