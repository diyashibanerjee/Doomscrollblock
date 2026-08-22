import cv2
import time
import winsound
import threading
from ultralytics import YOLO
import mediapipe as mp
import pygetwindow as gw

model = YOLO("yolov8n.pt")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

WORK_KEYWORDS = ["visual studio code", "code -"]
ALERT_THRESHOLD_SECONDS = 5
GRACE_PERIOD_SECONDS = 1.5  # how long we tolerate a "lost" hand before resetting the streak

phone_hold_start = None
last_detected_time = 0
already_alerted = False


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def is_work_mode():
    try:
        active_window = gw.getActiveWindow()
        if active_window is None:
            return False
        title = active_window.title.lower()
        return any(keyword in title for keyword in WORK_KEYWORDS)
    except Exception:
        return False


def play_alarm():
    # Runs in a separate thread so it doesn't freeze the video loop while playing.
    winsound.Beep(1000, 5000)  # 1000 Hz tone for 5000 ms = 5 seconds


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    work_mode = is_work_mode()

    results = model(frame, verbose=False)
    phone_boxes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id] == "cell phone" and float(box.conf[0]) > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phone_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    holding_phone_now = False
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_ids_to_check = [0, 9, 4, 8, 12, 16, 20]
            for lm_id in landmark_ids_to_check:
                lm = hand_landmarks.landmark[lm_id]
                px, py = int(lm.x * w), int(lm.y * h)
                for box in phone_boxes:
                    if point_in_box(px, py, box):
                        holding_phone_now = True

    # --- Debounce with grace period ---
    if holding_phone_now:
        last_detected_time = time.time()
        if phone_hold_start is None:
            phone_hold_start = time.time()
        elapsed = time.time() - phone_hold_start
    else:
        if phone_hold_start is not None and (time.time() - last_detected_time) < GRACE_PERIOD_SECONDS:
            # Still within the grace window — treat the streak as ongoing,
            # just don't update last_detected_time since we didn't actually see it this frame.
            elapsed = time.time() - phone_hold_start
        else:
            phone_hold_start = None
            elapsed = 0
            already_alerted = False

    status = f"Work mode: {work_mode} | Phone in hand: {holding_phone_now} | Held: {elapsed:.1f}s"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if elapsed >= ALERT_THRESHOLD_SECONDS:
        cv2.putText(frame, "STOP DOOMSCROLLING!", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if not already_alerted:
            print("ALERT: Doomscrolling detected during work session!")
            threading.Thread(target=play_alarm, daemon=True).start()
            already_alerted = True

    cv2.imshow("Focus Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
