
import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

URL = "http://192.168.4.1:81/stream"

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

print("Connecting to ESP32-CAM...")

while True:
    try:
        req = urllib.request.Request(
            URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            print("Connected:", response.status)

            buffer = b""

            while True:
                chunk = response.read(4096)
                if not chunk:
                    break

                buffer += chunk

                # Extract complete JPEG frames using JPEG markers
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)

                if start != -1 and end != -1:
                    jpeg = buffer[start:end + 2]
                    buffer = buffer[end + 2:]

                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        continue

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[2, 3, 5, 7],
                        conf=0.4,
                        verbose=False
                    )

                    annotated = results[0].plot()
                    cv2.imshow(
                        "ESP32-CAM YOLO Traffic Detection",
                        annotated
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise SystemExit

                    # Prevent incomplete data growing indefinitely
                    if len(buffer) > 2_000_000:
                        buffer = b""

    except SystemExit:
        break

    except Exception as e:
        print("Stream error:", e)
        print("Retrying in 2 seconds...")
        import time
        time.sleep(2)

cv2.destroyAllWindows()
