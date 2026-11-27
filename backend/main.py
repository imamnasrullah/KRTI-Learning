import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymavlink import mavutil
from fastapi.responses import StreamingResponse
from gz.transport import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import cv2
import time

app = FastAPI(title="KRTI Virtual Drone Laboratory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


telemetry = {
    "connected": False,
    "armed": False,
    "mode": "UNKNOWN",
    "altitude": 0,
    "relative_altitude": 0,
    "ground_speed": 0,
    "battery": 0,
    "latitude": 0,
    "longitude": 0,
    "message": "Menunggu koneksi ArduPilot..."
}

camera_lock = threading.Lock()
latest_jpeg = None


def gazebo_camera_listener():
    global latest_jpeg

    node = Node()

    def camera_callback(msg):
        global latest_jpeg

        try:
            if msg.pixel_format_type != 3:
                return

            image = np.frombuffer(
                msg.data, dtype=np.uint8
            ).reshape(msg.height, msg.step)

            image = image[:, :msg.width * 3].reshape(
                msg.height, msg.width, 3
            )

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", image)

            if success:
                with camera_lock:
                    latest_jpeg = encoded.tobytes()

        except Exception as error:
            print("Camera error:", error)

    node.subscribe(Image, "/camera/front/image", camera_callback)
    print("Gazebo front camera berhasil disubscribe!")

    while True:
        time.sleep(1)


def video_stream():
    while True:
        with camera_lock:
            frame = latest_jpeg

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )

        time.sleep(0.05)


def mavlink_listener():
    try:
        master = mavutil.mavlink_connection(
            "udpin:0.0.0.0:14550"
        )

        print("Menunggu heartbeat ArduPilot...")
        master.wait_heartbeat(timeout=60)

        telemetry["connected"] = True
        telemetry["message"] = "ArduPilot terhubung"
        print("ArduPilot berhasil terhubung!")

        while True:
            msg = master.recv_match(blocking=True, timeout=1)

            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "HEARTBEAT":
                telemetry["armed"] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                telemetry["mode"] = mavutil.mode_string_v10(msg)

            elif msg_type == "GLOBAL_POSITION_INT":
                telemetry["altitude"] = round(msg.alt / 1000, 2)
                telemetry["relative_altitude"] = round(
                    msg.relative_alt / 1000, 2
                )
                telemetry["latitude"] = msg.lat / 1e7
                telemetry["longitude"] = msg.lon / 1e7

            elif msg_type == "VFR_HUD":
                telemetry["ground_speed"] = round(msg.groundspeed, 2)

            elif msg_type == "SYS_STATUS":
                if msg.battery_remaining >= 0:
                    telemetry["battery"] = msg.battery_remaining

    except Exception as error:
        telemetry["connected"] = False
        telemetry["message"] = str(error)
        print("MAVLink error:", error)


@app.on_event("startup")
async def startup():
    thread = threading.Thread(
        target=mavlink_listener,
        daemon=True
    )
    thread.start()

    camera_thread = threading.Thread(
        target=gazebo_camera_listener,
        daemon=True
    )
    camera_thread.start()


@app.get("/")
async def home():
    return {
        "status": "KRTI Backend aktif",
        "telemetry": telemetry
    }


@app.get("/api/telemetry")
async def get_telemetry():
    return telemetry


@app.get("/video/front.mjpg")
async def front_camera():
    return StreamingResponse(
        video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            await websocket.send_json(telemetry)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
