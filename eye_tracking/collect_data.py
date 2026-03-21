#!/usr/bin/env python3
"""
Eye-tracking threshold data collector.

Run from the project root:
    python -m eye_tracking.collect_data

Controls:
    O  –  label current frame as ON SCREEN  (eyes open, looking at camera)
    A  –  label current frame as LOOKING AWAY  (eyes open, gaze off to side)
    C  –  label current frame as EYES CLOSED
    Q  –  quit and print per-label statistics

Output CSV columns:
    timestamp, left_ear, right_ear, avg_ear,
    left_gaze, right_gaze, avg_gaze, label
"""
import csv
import os
import statistics
import time

import cv2
import mediapipe as mp
import numpy as np

from eye_tracking.utils import (
    eye_aspect_ratio, gaze_ratio,
    LEFT_EYE, RIGHT_EYE,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_INNER, LEFT_EYE_OUTER,
    RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_DIR, "face_landmarker.task")
OUTPUT_CSV = os.path.join(_DIR, "eye_data.csv")

# ── MediaPipe (IMAGE mode = synchronous) ─────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
)

IRIS_INDICES = [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]

# ── CSV ───────────────────────────────────────────────────────────────────────
csv_file = open(OUTPUT_CSV, "a", newline="")
writer = csv.writer(csv_file)
if os.path.getsize(OUTPUT_CSV) == 0:
    writer.writerow([
        "timestamp",
        "left_ear", "right_ear", "avg_ear",
        "left_gaze", "right_gaze", "avg_gaze",
        "label",
    ])


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary():
    cols = ["left_ear", "right_ear", "avg_ear", "left_gaze", "right_gaze", "avg_gaze"]
    labels = ["on", "away", "closed"]
    data = {lbl: {c: [] for c in cols} for lbl in labels}

    try:
        with open(OUTPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                lbl = row.get("label", "").strip()
                if lbl not in data:
                    continue
                for c in cols:
                    try:
                        data[lbl][c].append(float(row[c]))
                    except (ValueError, KeyError):
                        pass
    except FileNotFoundError:
        return

    def s(vals):
        if not vals:
            return "n/a"
        sv = sorted(vals)
        p5  = sv[max(0, int(len(sv) * 0.05))]
        p95 = sv[int(len(sv) * 0.95)]
        return (f"n={len(vals):3d}  "
                f"min={min(vals):.3f}  mean={statistics.mean(vals):.3f}  "
                f"max={max(vals):.3f}  p5={p5:.3f}  p95={p95:.3f}")

    print("\n" + "=" * 70)
    print("EYE-TRACKING DATA SUMMARY")
    print("=" * 70)

    for metric in cols:
        print(f"\n{metric.upper()}")
        for lbl in labels:
            print(f"  {lbl:8s}: {s(data[lbl][metric])}")

    # ── Suggested thresholds ─────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Suggested thresholds:")

    # EAR_THRESHOLD: midpoint between closed p95 and on p5
    closed_ear = data["closed"]["avg_ear"]
    on_ear = data["on"]["avg_ear"]
    if closed_ear and on_ear:
        sv_closed = sorted(closed_ear)
        sv_on = sorted(on_ear)
        closed_p95 = sv_closed[int(len(sv_closed) * 0.95)]
        on_p5 = sv_on[max(0, int(len(sv_on) * 0.05))]
        ear_thresh = round((closed_p95 + on_p5) / 2, 3)
        print(f"  EAR_THRESHOLD  = {ear_thresh}"
              f"   (closed p95={closed_p95:.3f}, on p5={on_p5:.3f})")
    else:
        print("  EAR_THRESHOLD  = insufficient data")

    # GAZE_THRESHOLD: uses deviation = min(gaze, 1-gaze)
    # on-screen: deviation clusters near 0.5 (centred iris)
    # away:      deviation clusters near 0 (iris at edge)
    # threshold sits between on p5-deviation and away p95-deviation
    on_gaze = data["on"]["avg_gaze"]
    away_gaze = data["away"]["avg_gaze"]
    if on_gaze and away_gaze:
        on_dev = sorted(min(g, 1 - g) for g in on_gaze)
        away_dev = sorted(min(g, 1 - g) for g in away_gaze)
        on_p5_dev = on_dev[max(0, int(len(on_dev) * 0.05))]
        away_p95_dev = away_dev[int(len(away_dev) * 0.95)]
        gaze_thresh = round((on_p5_dev + away_p95_dev) / 2, 3)
        print(f"  GAZE_THRESHOLD = {gaze_thresh}"
              f"   (on p5-dev={on_p5_dev:.3f}, away p95-dev={away_p95_dev:.3f})")
    else:
        print("  GAZE_THRESHOLD = insufficient data")

    print()
    print("Copy these into eye_tracking/eye_tracking.py to tune your thresholds.")
    print("=" * 70)


# ── HUD helper ────────────────────────────────────────────────────────────────
def put(frame, text, y, color=(255, 255, 255), scale=0.55, bold=False):
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                2 if bold else 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
counts = {"on": 0, "away": 0, "closed": 0}
flash = None   # (label, expiry_time)

print(f"Output: {OUTPUT_CSV}")
print("O = on screen  |  A = looking away  |  C = eyes closed  |  Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    lms = result.face_landmarks[0] if result.face_landmarks else None

    l_ear = r_ear = avg_ear = None
    l_gaze = r_gaze = avg_gaze = None
    ready = False

    if lms:
        l_ear = eye_aspect_ratio(lms, LEFT_EYE, w, h)
        r_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
        avg_ear = (l_ear + r_ear) / 2.0

        l_gaze = gaze_ratio(lms, LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER)
        r_gaze = gaze_ratio(lms, RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
        avg_gaze = (l_gaze + r_gaze) / 2.0

        ready = True

        # Draw iris dots
        for idx in IRIS_INDICES:
            cx, cy = int(lms[idx].x * w), int(lms[idx].y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)

        # Draw eye outlines
        for eye_pts in [LEFT_EYE, RIGHT_EYE]:
            pts = np.array(
                [(int(lms[i].x * w), int(lms[i].y * h)) for i in eye_pts],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    # ── HUD ──────────────────────────────────────────────────────────────────
    put(frame, "EYE-TRACKING DATA COLLECTOR", 22, (0, 220, 220), scale=0.65, bold=True)

    y = 50
    if ready:
        put(frame, f"Left EAR  : {l_ear:.3f}    Right EAR  : {r_ear:.3f}    Avg EAR  : {avg_ear:.3f}", y)
        y += 20
        put(frame, f"Left gaze : {l_gaze:.3f}    Right gaze : {r_gaze:.3f}    Avg gaze : {avg_gaze:.3f}", y)
        y += 20
        # deviation = how centred the gaze is (0 = at edge, 0.5 = perfectly centred)
        dev = min(avg_gaze, 1 - avg_gaze)
        put(frame, f"Gaze deviation from edge: {dev:.3f}  (lower = more off-centre)", y, (180, 180, 255))
    else:
        put(frame, "Waiting for face...", y, (80, 80, 255))

    y += 26
    status_col = (120, 255, 120) if ready else (80, 80, 255)
    put(frame, "Ready to label" if ready else "No face detected", y, status_col)

    y += 20
    put(frame, f"On: {counts['on']}   Away: {counts['away']}   Closed: {counts['closed']}   Total: {sum(counts.values())}", y, (190, 190, 190))

    if flash and time.time() < flash[1]:
        col_map = {"on": (40, 200, 40), "away": (40, 40, 220), "closed": (200, 120, 40)}
        put(frame, f"SAVED: {flash[0].upper()}", frame.shape[0] - 45,
            col_map[flash[0]], scale=0.75, bold=True)

    put(frame, "[O] On screen  [A] Away  [C] Closed  [Q] Quit", frame.shape[0] - 18, (160, 160, 160))

    cv2.imshow("Eye-Tracking Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    label = None
    if key == ord('o') and ready:
        label = "on"
    elif key == ord('a') and ready:
        label = "away"
    elif key == ord('c') and ready:
        label = "closed"

    if label:
        writer.writerow([
            round(time.time(), 3),
            round(l_ear, 3), round(r_ear, 3), round(avg_ear, 3),
            round(l_gaze, 3), round(r_gaze, 3), round(avg_gaze, 3),
            label,
        ])
        csv_file.flush()
        counts[label] += 1
        flash = (label, time.time() + 1.2)

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print_summary()
