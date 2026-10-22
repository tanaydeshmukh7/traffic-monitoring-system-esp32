import os

import time

import threading

import queue

from datetime import datetime

from collections import deque

import cv2

import numpy as np

import requests

try:
    import serial
except ImportError:
    serial = None

from flask import Flask, Response, jsonify, request, send_from_directory

from ultralytics import YOLO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# CONFIGURATION
CAMERA_URL = os.getenv("CAMERA_URL", "http://10.153.74.248:81/stream").strip()

MODEL_PATH = "yolo11n.pt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

CONFIDENCE = 0.35

IMAGE_SIZE = 320

PROCESS_WIDTH = 480

PROCESS_HEIGHT = 360

CAPTURE_COOLDOWN = 3

RECONNECT_DELAY = 3

MAX_EVENTS = 100

ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM15").strip()
ARDUINO_BAUD = int(os.getenv("ARDUINO_BAUD", "9600"))
GREEN_DURATION = float(os.getenv("GREEN_DURATION", "30"))
RED_DURATION = float(os.getenv("RED_DURATION", "30"))
# Telegram credentials are provided through environment variables.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

os.makedirs(EVIDENCE_DIR, exist_ok=True)

VEHICLES = {

    2: "Car",

    3: "Motorcycle",

    5: "Bus",

    7: "Truck",

}

app = Flask(__name__)

model = YOLO(MODEL_PATH)
# SHARED STATE
frame_lock = threading.Lock()

data_lock = threading.Lock()

config_lock = threading.Lock()

latest_frame = None

frame_version = 0

latest_jpeg = None

events = deque(maxlen=MAX_EVENTS)

line_config = {

    "orientation": "vertical",

    "position": 50,

}

camera_status = {
    "connected": False,
    "message": "Starting camera",
}

signal_lock = threading.Lock()
signal_status = {
    "state": "GREEN",
    "phase_started": time.monotonic(),
    "seconds_remaining": GREEN_DURATION,
    "arduino_connected": False,
    "message": "Waiting for Arduino serial connection",
}
previous_signal_state = {}
previous_signal_phase = {}
# Keyed by tracker ID and vehicle class.
previous_positions = {}

previous_line_overlap = {}

last_capture = {}
# TELEGRAM INTEGRATION
telegram_queue = queue.Queue(maxsize=100)

def send_telegram_photo(photo_path, vehicle, track_id, timestamp):

    """Queue an evidence photo so Telegram networking never blocks YOLO."""

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:

        print("[Telegram] Not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

        return

    try:

        telegram_queue.put_nowait((photo_path, vehicle, track_id, timestamp))

        print(f"[Telegram] Queued photo: {os.path.basename(photo_path)}")

    except queue.Full:

        print("[Telegram] Queue full; photo remains saved locally:", photo_path)

def telegram_worker():

    """Send queued photos with retries for temporary DNS/network failures."""

    while True:

        item = telegram_queue.get()

        photo_path, vehicle, track_id, timestamp = item

        caption = (

            "🚨 Traffic line-crossing alert\n"

            f"Vehicle: {vehicle}\n"

            f"Track ID: {track_id}\n"

            f"Time: {timestamp}"

        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        delivered = False

        try:

            for attempt in range(1, 7):

                try:

                    with open(photo_path, "rb") as photo:

                        response = requests.post(

                            url,

                            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},

                            files={"photo": (os.path.basename(photo_path), photo, "image/jpeg")},

                            timeout=(10, 40),

                        )

                    try:

                        result = response.json()

                    except ValueError:

                        result = {"description": response.text[:500]}

                    if response.ok and result.get("ok"):

                        print("[Telegram] Photo sent successfully:", photo_path)

                        delivered = True

                        break
# Telegram API replied, so print its actual error instead of retrying forever.
                    print(f"[Telegram] API rejected photo (HTTP {response.status_code}): {result}")

                    if response.status_code not in (429, 500, 502, 503, 504):

                        break

                except (requests.RequestException, OSError) as exc:

                    print(f"[Telegram] Network/file error on attempt {attempt}/6: {exc}")

                if attempt < 6:

                    delay = min(60, 2 ** attempt)

                    print(f"[Telegram] Retrying in {delay}s; evidence is saved locally.")

                    time.sleep(delay)

            if not delivered:

                print("[Telegram] Could not deliver after retries. Evidence remains in:", os.path.abspath(photo_path))

        finally:

            telegram_queue.task_done()
# MOBILE DASHBOARD
@app.route("/")

def dashboard():

    return """

<!DOCTYPE html>

<html>

<head>

<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Traffic Monitoring System</title>

<style>

body {

    margin: 0; padding: 16px; font-family: Arial, sans-serif;

    background: #101827; color: #f8fafc;

}

.panel {

    background: #1e293b; padding: 15px;

    margin: 14px 0; border-radius: 12px;

}

img { width: 100%; border-radius: 8px; background: #000; }

select, input, button {

    width: 100%; box-sizing: border-box;

    padding: 12px; margin: 8px 0; border-radius: 8px;

}

button {

    background: #22c55e; color: #052e16;

    border: 0; font-weight: bold;

}

.event {

    background: #334155; padding: 12px;

    margin: 10px 0; border-radius: 8px;

    overflow-wrap: anywhere;

}

a { color: #93c5fd; }

small { color: #cbd5e1; }

</style>

</head>

<body>

<h2>Traffic Monitoring System</h2>

<small>ESP32-CAM + YOLO</small>

<div class="panel">

    <h3>Live Detection</h3>

    <img src="/video.mjpg" alt="Waiting for camera">

    <p id="status">Starting...</p>

    <p id="fps">Detection FPS: measuring...</p>

</div>

<div class="panel">

    <h3>Virtual Line Settings</h3>

    <label>Orientation</label>

    <select id="orientation">

        <option value="horizontal">Horizontal</option>

        <option value="vertical">Vertical</option>

    </select>

    <label>Position: <span id="positionLabel">50</span>%</label>

    <input id="position" type="range" min="5" max="95" value="50">

    <button onclick="saveConfig()">Apply Settings</button>

</div>

<div class="panel">

    <h3>Crossing Evidence</h3>

    <p id="count">0 events</p>

    <div id="events">No crossings detected yet</div>

</div>

<script>

const orientation = document.getElementById("orientation");

const position = document.getElementById("position");

position.addEventListener("input", () => {

    document.getElementById("positionLabel").textContent = position.value;

});

async function loadConfig() {

    try {

        const r = await fetch("/config");

        const c = await r.json();

        orientation.value = c.orientation;

        position.value = c.position;

        document.getElementById("positionLabel").textContent = c.position;

    } catch (e) {}

}

async function saveConfig() {

    try {

        const r = await fetch("/config", {

            method: "POST",

            headers: {"Content-Type": "application/json"},

            body: JSON.stringify({

                orientation: orientation.value,

                position: Number(position.value)

            })

        });

        const result = await r.json();

        document.getElementById("status").textContent = result.message;

    } catch (e) {

        document.getElementById("status").textContent = "Settings update failed";

    }

}

async function refreshEvents() {

    try {

        const r = await fetch("/events");

        const data = await r.json();

        document.getElementById("count").textContent = data.length + " recent events";

        const container = document.getElementById("events");

        container.replaceChildren();

        if (!data.length) {

            container.textContent = "No crossings detected yet";

            return;

        }

        for (const e of data) {

            const card = document.createElement("div");

            card.className = "event";

            const title = document.createElement("b");

            title.textContent = e.vehicle;

            const details = document.createElement("p");

            details.textContent = "Time: " + e.time + " | Track ID: " + e.track_id;

            const link = document.createElement("a");

            link.href = "/evidence/" + encodeURIComponent(e.photo);

            link.target = "_blank";

            link.textContent = "View evidence photo";

            card.append(title, details, link);

            container.appendChild(card);

        }

    } catch (e) {

        document.getElementById("events").textContent = "Unable to load events";

    }

}

async function refreshStatus() {

    try {

        const r = await fetch("/status");

        const s = await r.json();

        document.getElementById("status").textContent = s.camera + " | Signal: " + (s.signal || "UNKNOWN") + " (" + (s.seconds_remaining ?? "-") + "s)" + (s.arduino_connected ? " | Arduino connected" : " | Arduino disconnected");

        document.getElementById("fps").textContent = "Detection FPS: " + s.fps;

    } catch (e) {}

}

loadConfig();

refreshEvents();

refreshStatus();

setInterval(refreshEvents, 2000);

setInterval(refreshStatus, 1000);

</script>

</body>

</html>

"""

@app.route("/config", methods=["GET", "POST"])

def config_endpoint():

    with config_lock:

        if request.method == "POST":

            data = request.get_json(silent=True) or {}

            if data.get("orientation") in ("horizontal", "vertical"):

                line_config["orientation"] = data["orientation"]

            try:

                if "position" in data:

                    line_config["position"] = max(

                        5, min(95, int(data["position"]))

                    )

            except (ValueError, TypeError):

                pass

            return jsonify({

                **line_config,

                "message": "Virtual line updated",

            })

        return jsonify(line_config.copy())

@app.route("/events")

def get_events():

    with data_lock:

        return jsonify(list(events))

@app.route("/api/status")
@app.route("/status")
def get_status():
    with data_lock:
        connected = camera_status["connected"]
        message = camera_status["message"]
        fps = detection_fps
    with signal_lock:
        signal = signal_status["state"]
        remaining = max(0, int(signal_status["seconds_remaining"] + 0.999))
        arduino_connected = signal_status["arduino_connected"]
        arduino_message = signal_status["message"]
    return jsonify({
        "camera": "Camera connected" if connected else message,
        "fps": round(fps, 1),
        "signal": signal,
        "seconds_remaining": remaining,
        "arduino_connected": arduino_connected,
        "arduino_message": arduino_message,
    })

@app.route("/evidence/<path:filename>")

def get_evidence(filename):

    return send_from_directory(EVIDENCE_DIR, filename)

@app.route("/video.mjpg")

def video():

    def generate():

        last_sent = None

        while True:

            with data_lock:

                jpeg = latest_jpeg

            if jpeg is not None and jpeg is not last_sent:

                last_sent = jpeg

                yield (

                    b"--frame\r\n"

                    b"Content-Type: image/jpeg\r\n"

                    b"Content-Length: " + str(len(jpeg)).encode()

                    + b"\r\n\r\n" + jpeg + b"\r\n"

                )

            time.sleep(0.03)

    return Response(

        generate(),

        mimetype="multipart/x-mixed-replace; boundary=frame",

        headers={"Cache-Control": "no-cache, no-store"},

    )
# NON-BLOCKING SIGNAL CONTROLLER

def signal_controller():
    """Run timed phases without blocking the camera, YOLO, Flask, or Telegram."""
    ser = None
    last_sent_state = None
    state = "GREEN"
    phase_started = time.monotonic()

    def send_state(new_state):
        nonlocal ser, last_sent_state
        if serial is None:
            with signal_lock:
                signal_status["arduino_connected"] = False
                signal_status["message"] = "pyserial missing; run pip install pyserial"
            return
        if ser is None or not getattr(ser, "is_open", False):
            try:
                ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.2, write_timeout=1)
                time.sleep(2.0)  # Arduino Uno resets when serial opens.
                with signal_lock:
                    signal_status["arduino_connected"] = True
                    signal_status["message"] = f"Arduino connected on {ARDUINO_PORT}"
                last_sent_state = None
            except Exception as exc:
                ser = None
                with signal_lock:
                    signal_status["arduino_connected"] = False
                    signal_status["message"] = f"Arduino serial unavailable: {exc}"
                return
        if last_sent_state != new_state:
            try:
                ser.write((new_state + "\n").encode("ascii"))
                ser.flush()
                last_sent_state = new_state
                print(f"[Signal] Sent {new_state} to Arduino")
            except Exception as exc:
                print("[Signal] Serial write failed:", exc)
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                with signal_lock:
                    signal_status["arduino_connected"] = False
                    signal_status["message"] = f"Arduino serial write failed: {exc}"

    while True:
        now = time.monotonic()
        duration = GREEN_DURATION if state == "GREEN" else RED_DURATION
        if now - phase_started >= duration:
            state = "RED" if state == "GREEN" else "GREEN"
            phase_started = now
            print(f"[Signal] Phase changed to {state}")
            duration = GREEN_DURATION if state == "GREEN" else RED_DURATION
        send_state(state)
        remaining = max(0.0, duration - (now - phase_started))
        with signal_lock:
            signal_status["state"] = state
            signal_status["phase_started"] = phase_started
            signal_status["seconds_remaining"] = remaining
        time.sleep(0.1)


# CAMERA READER THREAD
def camera_reader():

    global latest_frame, frame_version

    while True:

        try:

            print("Connecting to ESP32-CAM...")

            with requests.get(

                CAMERA_URL,

                stream=True,

                timeout=(5, 30),

                headers={"User-Agent": "TrafficMonitor"},

            ) as response:

                response.raise_for_status()

                print("Camera connected:", response.status_code)

                with data_lock:

                    camera_status["connected"] = True

                    camera_status["message"] = "Camera connected"

                buffer = bytearray()

                for chunk in response.iter_content(chunk_size=1024):

                    if not chunk:

                        continue

                    buffer.extend(chunk)

                    while True:

                        start = buffer.find(b"\xff\xd8")

                        if start == -1:

                            if len(buffer) > 1:

                                del buffer[:-1]

                            break

                        end = buffer.find(b"\xff\xd9", start + 2)

                        if end == -1:

                            if start > 0:

                                del buffer[:start]

                            break

                        jpeg = bytes(buffer[start:end + 2])

                        del buffer[:end + 2]

                        frame = cv2.imdecode(

                            np.frombuffer(jpeg, dtype=np.uint8),

                            cv2.IMREAD_COLOR,

                        )

                        if frame is not None:

                            with frame_lock:

                                latest_frame = frame

                                frame_version += 1

                    if len(buffer) > 2_000_000:

                        buffer.clear()

        except Exception as exc:

            print("Camera stream error:", repr(exc))

            with data_lock:

                camera_status["connected"] = False

                camera_status["message"] = "Reconnecting to camera"

            time.sleep(RECONNECT_DELAY)
# YOLO + LINE CROSSING + EVIDENCE
detection_fps = 0.0

def run_detection():

    global latest_jpeg, detection_fps

    processed_version = -1

    fps_count = 0

    fps_start = time.monotonic()

    while True:

        with frame_lock:

            if latest_frame is not None and frame_version != processed_version:

                frame = latest_frame.copy()

                current_version = frame_version

            else:

                frame = None

                current_version = processed_version

        if frame is None:

            time.sleep(0.005)

            continue

        processed_version = current_version

        try:

            height, width = frame.shape[:2]
# Resize before YOLO to reduce CPU load.
            if width > PROCESS_WIDTH:

                new_height = int(height * PROCESS_WIDTH / width)

                frame = cv2.resize(

                    frame, (PROCESS_WIDTH, new_height),

                    interpolation=cv2.INTER_AREA,

                )

            height, width = frame.shape[:2]

            with config_lock:

                orientation = line_config["orientation"]

                position = line_config["position"]

            line_coord = int(

                (height if orientation == "horizontal" else width)

                * position / 100

            )

            result = model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                classes=list(VEHICLES.keys()),

                conf=CONFIDENCE,

                imgsz=IMAGE_SIZE,

                max_det=20,

                verbose=False,

            )[0]

            annotated = result.plot()
            with signal_lock:
                current_signal = signal_status["state"]
                current_signal_phase = signal_status["phase_started"]
                signal_remaining = max(0, int(signal_status["seconds_remaining"] + 0.999))

            # Only show the virtual stop line during RED.
            if current_signal == "RED" and orientation == "horizontal":
                cv2.line(annotated, (0, line_coord), (width, line_coord), (0, 0, 255), 2)
            elif current_signal == "RED":
                cv2.line(annotated, (line_coord, 0), (line_coord, height), (0, 0, 255), 2)

            signal_color = (0, 0, 255) if current_signal == "RED" else (0, 220, 0)
            cv2.putText(annotated, f"SIGNAL: {current_signal} | {signal_remaining}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, signal_color, 2)
            if current_signal == "RED":
                cv2.putText(annotated, f"LINE: {orientation.upper()} {position}%",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:

                coords = boxes.xyxy.cpu().tolist()

                classes = boxes.cls.int().cpu().tolist()

                # ByteTrack IDs can be None for a frame or two. Do not silently
                # disable evidence capture in that case; use temporary per-frame
                # keys as a fallback. Stable IDs are preferred whenever available.
                if boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                else:
                    ids = [-(index + 1) for index in range(len(coords))]
                    print("[YOLO] Tracking IDs unavailable this frame; using fallback IDs")

                for track_id, box, cls in zip(ids, coords, classes):

                    if cls not in VEHICLES:

                        continue

                    x1, y1, x2, y2 = map(int, box)

                    cx = (x1 + x2) // 2

                    cy = (y1 + y2) // 2

                    coord = cy if orientation == "horizontal" else cx

                    key = (track_id, cls)

                    old_coord = previous_positions.get(key)

                    # Count a crossing when the center changes sides. Inclusive
                    # comparison handles a center that lands exactly on the line.
                    center_crossed = (
                        old_coord is not None
                        and (
                            (old_coord < line_coord and coord >= line_coord)
                            or (old_coord > line_coord and coord <= line_coord)
                        )
                    )

                    if orientation == "horizontal":

                        touching = y1 <= line_coord <= y2

                    else:

                        touching = x1 <= line_coord <= x2

                    was_touching = previous_line_overlap.get(key, False)
# Trigger on first contact or center crossing.
                    crossed = ((touching and not was_touching) or center_crossed)
                    # Decide using the signal state at the moment the crossing
                    # is observed. Requiring the previous observation to be in
                    # the exact same red phase can miss real crossings near a
                    # phase transition or after tracker history is refreshed.
                    # Green-phase crossings are never captured.
                    eligible_red_crossing = (current_signal == "RED" and crossed)
                    if crossed:
                        print(
                            f"[Crossing] {VEHICLES[cls]} ID={track_id} "
                            f"signal={current_signal} line={orientation}:{position}% "
                            f"center=({cx},{cy}) touching={touching} "
                            f"eligible={eligible_red_crossing}"
                        )

                    now = time.monotonic()

                    last_time = last_capture.get(key, 0)

                    if eligible_red_crossing and now - last_time >= CAPTURE_COOLDOWN:

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        filename = datetime.now().strftime(

                            "crossing_%Y%m%d_%H%M%S_%f.jpg"

                        )
# Save a copy so annotations don't affect other detections.
                        evidence = annotated.copy()

                        cv2.rectangle(

                            evidence, (x1, y1), (x2, y2),

                            (0, 255, 255), 3,

                        )

                        cv2.putText(

                            evidence,

                            f"{VEHICLES[cls]} ID {track_id} | {timestamp}",

                            (10, min(height - 10, 55)),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.55, (0, 255, 255), 2,

                        )

                        photo_path = os.path.join(EVIDENCE_DIR, filename)

                        if cv2.imwrite(photo_path, evidence):

                            print(f"[Evidence] SAVED: {os.path.abspath(photo_path)}")

                            event = {

                                "vehicle": VEHICLES[cls],

                                "track_id": int(track_id),

                                "time": timestamp,

                                "photo": filename,

                            }

                            with data_lock:

                                events.appendleft(event)

                            last_capture[key] = now

                            print("Evidence captured:", event)
# Telegram integration: send the same saved evidence.
                            send_telegram_photo(

                                photo_path,

                                VEHICLES[cls],

                                track_id,

                                timestamp,

                            )
                        else:
                            print(
                                f"[Evidence] ERROR: cv2.imwrite returned False. "
                                f"Path={os.path.abspath(photo_path)} "
                                f"directory_exists={os.path.isdir(EVIDENCE_DIR)} "
                                f"frame_shape={evidence.shape}"
                            )

                    previous_positions[key] = coord

                    previous_line_overlap[key] = touching
                    previous_signal_state[key] = current_signal
                    previous_signal_phase[key] = current_signal_phase
# Publish latest annotated frame.
            ok, encoded = cv2.imencode(

                ".jpg", annotated,

                [cv2.IMWRITE_JPEG_QUALITY, 75],

            )

            if ok:

                with data_lock:

                    latest_jpeg = encoded.tobytes()

            fps_count += 1

            elapsed = time.monotonic() - fps_start

            if elapsed >= 1.0:

                detection_fps = fps_count / elapsed

                fps_count = 0

                fps_start = time.monotonic()
# Bound tracking memory.
            if len(previous_positions) > 1000:

                previous_positions.clear()
                previous_line_overlap.clear()
                previous_signal_state.clear()
                previous_signal_phase.clear()
                last_capture.clear()

        except Exception as exc:

            print("YOLO detection error:", repr(exc))

            time.sleep(0.05)
# START APPLICATION
if __name__ == "__main__":

    if not TELEGRAM_BOT_TOKEN:

        print(

            "WARNING: TELEGRAM_BOT_TOKEN is not set. "

            "Evidence will still be saved locally, but Telegram sending is disabled."

        )

    threading.Thread(

        target=lambda: app.run(

            host="0.0.0.0",

            port=5000,

            debug=False,

            use_reloader=False,

            threaded=True,

        ),

        daemon=True,

    ).start()

    threading.Thread(
        target=signal_controller,
        daemon=True,
        name="signal-controller",
    ).start()

    threading.Thread(
        target=camera_reader,
        daemon=True,
    ).start()

    threading.Thread(

        target=telegram_worker,

        daemon=True,

        name="telegram-worker",

    ).start()

    print("Dashboard: http://127.0.0.1:5000")

    print("Evidence directory:", os.path.abspath(EVIDENCE_DIR))

    print("Telegram configured:", bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID))

    print("Camera URL:", CAMERA_URL)
    print(f"Signal cycle: GREEN {GREEN_DURATION}s / RED {RED_DURATION}s")
    print("Arduino port:", ARDUINO_PORT)

    run_detection()
