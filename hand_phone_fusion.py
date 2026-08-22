import cv2
from ultralytics import YOLO
import mediapipe as mp

model = YOLO("yolov8n.pt")

# mp.solutions.hands is MediaPipe's pretrained hand-tracking model.
# It detects 21 landmark points per hand (knuckles, fingertips, wrist, etc).
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,   # False = optimized for video streams (tracks between frames, faster)
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils  # helper to draw landmarks on the frame, useful for debugging

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open webcam")
    exit()


def point_in_box(x, y, box):
    """Check if a single (x, y) point falls inside a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape  # frame's pixel height/width — we need this to convert
                            # MediaPipe's landmark coordinates (0.0 to 1.0) into real pixels

    # --- Step 1: find phone boxes with YOLO (same as before) ---
    results = model(frame, verbose=False)
    phone_boxes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id] == "cell phone" and float(box.conf[0]) > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phone_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- Step 2: find hands with MediaPipe ---
    # MediaPipe expects RGB images, but OpenCV gives us BGR by default — so we convert.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    holding_phone = False

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # landmark 9 is the base of the middle finger — a decent stand-in
            # for "center of the palm." Landmarks are normalized (0.0-1.0),
            # so we multiply by frame width/height to get real pixel coordinates.
            palm = hand_landmarks.landmark[9]
            px, py = int(palm.x * w), int(palm.y * h)

            for box in phone_boxes:
                if point_in_box(px, py, box):
                    holding_phone = True

    if holding_phone:
        cv2.putText(frame, "PHONE IN HAND", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print("Phone in hand detected!")

    cv2.imshow("Hand + Phone Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
