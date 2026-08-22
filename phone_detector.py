import cv2
from ultralytics import YOLO

# This downloads a small pretrained model (~6MB) the first time you run it,
# then caches it locally. "yolov8n.pt" = the "nano" version — smallest and
# fastest, good enough for our purposes and won't choke your laptop.
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on this single frame.
    # This returns a list of "Results" objects — one per image we passed in.
    # We only passed one frame, so results[0] is what we want.
    results = model(frame, verbose=False)

    # Each detected object is a "box" — it has a class id, confidence score,
    # and coordinates. We loop through all boxes found in this frame.
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]  # turns the id into a readable label, e.g. "cell phone"
        confidence = float(box.conf[0])

        if class_name == "cell phone" and confidence > 0.5:
            # xyxy gives us the box corners: top-left (x1,y1) and bottom-right (x2,y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Phone {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print("Phone detected!")

    cv2.imshow("Phone Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
