# 🛡️ DOOM-SCROLL-BLOCK

**A real-time computer vision system that catches you doomscrolling — and tells you to put the phone down.**

FocusGuard watches your webcam while you work, detects when you pick up your phone during a focus session, and fires an alert if you don't put it down. Built to actually solve the "I opened Instagram for 30 seconds and lost an hour" problem.

---

## ✨ What it does

- 📱 **Detects your phone** in the camera frame using a pretrained YOLOv8 object detection model
- ✋ **Confirms you're actually holding it** — not just that a phone is *somewhere* in frame — by fusing hand-tracking landmarks with the phone's bounding box
- 💻 **Checks you're supposed to be working** by monitoring the active window (e.g. VS Code)
- ⏱️ **Debounces detection** with a grace period, so a flicker or small hand movement doesn't reset the timer
- 🔊 **Sounds a real alarm** if the phone stays in your hand for too long during a work session
- 🗃️ **Logs every distraction event** to a local SQLite database — timestamp and duration — for later analysis




📷 Webcam Frame
│
▼
🎯 YOLOv8 → is a phone visible? → bounding box
│
▼
✋ MediaPipe Hands → are hand landmarks inside that box? → holding_phone
│
▼
💻 Active window check → is VS Code focused? → work_mode
│
▼
⏳ holding_phone AND work_mode, sustained 5+ seconds
│
▼
🚨 Alarm + 🗃️ Log to SQLite
---

## 🧠 How it works

1. Every frame from the webcam is passed through **YOLOv8**, pretrained on COCO, which already recognizes "cell phone" as a class — no custom training needed.
2. **MediaPipe** tracks 21 hand landmarks per hand in the same frame. Several key points (wrist, palm, fingertips) are checked against the phone's bounding box — if any overlap, the phone is considered "in hand," not just "in frame."
3. The currently focused window is checked against a list of work-related keywords to determine if you're actually in a study/work session.
4. If the phone is held **and** you're in work mode, a timer starts. Brief tracking loss (under 1.5s) doesn't reset it — only genuinely putting the phone down does.
5. Once the hold crosses the threshold, a 5-second alarm plays (on a background thread, so the camera feed doesn't freeze) and the event is logged.

---

## 🛠️ Tech stack

| Component | Tool |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Hand tracking | MediaPipe |
| Camera capture | OpenCV |
| Active window detection | PyGetWindow |
| Alerts | winsound (threaded) |
| Storage | SQLite |
| Language | Python 3.11 |

---

## 🚀 Getting started

```bash
# clone the repo
git clone https://github.com/yourusername/focusguard.git
cd focusguard

# create and activate a virtual environment (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate

# install dependencies
pip install opencv-python ultralytics mediapipe==0.10.21 pygetwindow

# run it
python focus_detector.py
```

Press `q` in the camera window to quit.

---

## 📊 Example output

While running, the app overlays live status on screen:
Work mode: True | Phone in hand: True | Held: 3.2s

And once triggered:

🚨 STOP DOOMSCROLLING!





---

## 🔒 Privacy

All video processing happens **locally, in real time**. No frames are ever saved, stored, or uploaded — the camera feed exists only in memory for the split second it takes to run detection, then it's gone.

---

## 🔮 Future improvements

- [ ] Replace simple window-title check with process-level detection for more robust "work mode" classification
- [ ] Add a small dashboard to visualize distraction time per day from the logged data
- [ ] Support additional work-mode contexts (browser-based study sites, not just VS Code)
- [ ] Migrate to MediaPipe's newer Tasks API (HandLandmarker) as it stabilizes
- [ ] Package as a standalone `.exe` for easier use without a Python setup

---

## 📄 License

All rights reserved. See [LICENSE](./LICENSE) for details.

---

*Built as a portfolio project exploring real-time computer vision, model fusion, and behavior-based system design.*
