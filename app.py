import cv2
from posture_detection.posture_detection import detect_posture, draw_posture_detection
from phone_detection.phone_detection import detect_phone, draw_phone_detection
from eye_detection.eye_detection import detect_eyes, draw_eye_detection

cap = cv2.VideoCapture(0)

FRAME_W, FRAME_H = 633, 367
CAP_FPS = 5.0
cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def calculate_focus(posture_pct, eye_pct, phone_pct):
    if phone_pct == 100:
        return 0
    return int((0.6 * posture_pct) + (0.4 * eye_pct))

def draw_text_overlay(frame, posture_pct, posture_color, eye_pct, eye_color, phone_pct, phone_color, focus_pct, focus_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    phone_text = "Phone Detected" if phone_pct == 100 else "No Phone"
    lines = [
        (f"Good Posture: {posture_pct}%", posture_color),
        (f"Concentrated Gaze: {eye_pct}%", eye_color),
        (phone_text, phone_color),
        (f"Focus: {focus_pct}%", focus_color),
    ]
    y = FRAME_H - 20 - (len(lines) - 1) * 30
    for text, color in lines:
        bgr = hex_to_bgr(color) if isinstance(color, str) else color
        cv2.putText(frame, text, (10, y), font, scale, bgr, thickness, cv2.LINE_AA)
        y += 30

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(cv2.resize(frame, (FRAME_W, FRAME_H)), 1)
    detect_posture(frame)
    detect_eyes(frame)
    detect_phone(frame)

    frame, phone_color, phone_pct = draw_phone_detection(frame)
    frame, posture_color, posture_pct = draw_posture_detection(frame)
    frame, eye_color, eye_pct = draw_eye_detection(frame)

    focus_pct = calculate_focus(posture_pct, eye_pct, phone_pct)
    focus_color = "#009138" if focus_pct >= 50 else "#ad3a00"

    draw_text_overlay(frame, posture_pct, posture_color, eye_pct, eye_color, phone_pct, phone_color, focus_pct, focus_color)

    cv2.imshow("Distraction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
