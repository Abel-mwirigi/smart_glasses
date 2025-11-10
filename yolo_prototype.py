#!/usr/bin/env python3
"""
Realtime YOLO -> TTS prototype for smart-glasses style assistant.

Features:
- Tries yolo11n then falls back to yolov8n
- Direction: left / ahead / right
- Rough distance estimation from bbox height
- Debounced TTS with a queue thread
"""

import time
import threading
import platform
import subprocess
from collections import defaultdict, deque

import numpy as np
import cv2

# Try importing model API
from ultralytics import YOLO

# TTS: try pyttsx3, else macOS 'say', else print as fallback
try:
    import pyttsx3

    _tts_engine = pyttsx3.init()

    def speak_text(text):
        _tts_engine.say(text)
        _tts_engine.runAndWait()

except Exception:
    if platform.system() == "Darwin":  # macOS fallback

        def speak_text(text):
            subprocess.run(["say", text])

    else:

        def speak_text(text):
            print("[TTS]", text)


# -------------- CONFIG --------------
MODEL_PREFERRED = ["yolo11n.pt", "yolov8n.pt"]  # try v11 then v8
IMGSZ = 640  # resize for model (speed vs accuracy)
CONF_THRESH = 0.5  # min confidence
SPEAK_COOLDOWN = 4.0  # seconds before repeating same label/direction
MAX_LABELS_PER_ANN = 2  # max spoken labels per frame
FRAME_SKIP = 1  # process every Nth frame (1 = every frame)
SPEAK_PREFIX = ""  # e.g. "I see" or empty for shorter speech
# ------------------------------------

# Attempt to load model (try preferred names)
model = None
for candidate in MODEL_PREFERRED:
    try:
        model = YOLO(candidate)
        print(f"[INFO] Loaded model: {candidate}")
        break
    except Exception as e:
        print(f"[WARN] Could not load {candidate}: {e}")
if model is None:
    raise SystemExit(
        "No YOLO model found. Download a model (yolo11n.pt or yolov8n.pt) and retry."
    )

# TTS queue & worker
tts_queue = deque()
tts_lock = threading.Lock()


def tts_worker():
    while True:
        try:
            if tts_queue:
                with tts_lock:
                    text = tts_queue.popleft()
                speak_text(text)
            else:
                time.sleep(0.05)
        except Exception as e:
            print("[TTS ERROR]", e)
            time.sleep(0.5)


worker = threading.Thread(target=tts_worker, daemon=True)
worker.start()

# cooldown map to prevent repeats
last_spoken = defaultdict(lambda: 0.0)


def enqueue_speech(text, key=None):
    """Enqueue text for TTS. key used for cooldown dedupe."""
    now = time.time()
    dedupe_key = key if key else text
    if now - last_spoken[dedupe_key] >= SPEAK_COOLDOWN:
        with tts_lock:
            tts_queue.append(text)
        last_spoken[dedupe_key] = now


def direction_from_bbox(x1, x2, frame_w):
    # x center
    xc = (x1 + x2) / 2.0
    if xc < frame_w / 3.0:
        return "left"
    elif xc > 2 * frame_w / 3.0:
        return "right"
    else:
        return "ahead"


def estimate_distance_from_box(box_h, frame_h):
    # Very rough: larger box height => closer. We map fraction to qualitative distance
    frac = box_h / frame_h
    if frac > 0.45:
        return "very close"
    if frac > 0.25:
        return "close"
    if frac > 0.12:
        return "nearby"
    return "far"


# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera. Check webcam index or permissions.")

frame_idx = 0
print("Starting detection. Press 'q' in window to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] frame read failed")
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            # optional display to keep UI updated
            cv2.imshow("YOLO-TTS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Run inference (resize done by model call)
        results = model(frame, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            # optionally speak quiet environment facts if needed
            cv2.imshow("YOLO-TTS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # extract detections (conf, cls, xyxy)
        dets = []
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()  # x1,y1,x2,y2
            dets.append((conf, cls_id, xyxy))

        # sort by confidence desc
        dets.sort(key=lambda x: x[0], reverse=True)

        frame_h, frame_w = frame.shape[:2]
        spoken_this_frame = 0
        described_keys = set()

        for conf, cls_id, xyxy in dets[:10]:  # inspect top 10
            if spoken_this_frame >= MAX_LABELS_PER_ANN:
                break
            x1, y1, x2, y2 = map(float, xyxy)
            box_h = y2 - y1
            box_w = x2 - x1
            name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"

            direction = direction_from_bbox(x1, x2, frame_w)
            distance = estimate_distance_from_box(box_h, frame_h)
            # build short phrase
            phrase = f"{SPEAK_PREFIX} {name} {direction} {distance}".strip()
            # make a dedupe key: name+direction to avoid repeat same object repeatedly
            key = f"{name}:{direction}"
            if key in described_keys:
                continue
            # cooldown check handled in enqueue_speech
            enqueue_speech(phrase, key=key)
            described_keys.add(key)
            spoken_this_frame += 1

        # visualization
        annotated = r.plot()  # uses Ultralytics plot
        cv2.imshow("YOLO-TTS", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopping by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly")
