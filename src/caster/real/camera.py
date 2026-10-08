
from pathlib import Path
import select
import sys
import threading
import time

import numpy as np
from termcolor import colored
import viser

import cv2
import zmq


class CameraFeed:
    def __init__(self, address: str = "tcp://127.0.0.1:5555", preview_port: int = 8082):
        self.address = address
        self.preview_server = None
        self.preview_image = None
        self._preview_frame = None
        self._preview_updated_at = 0.0
        self.available = False
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.recording = False
        self.record_path = None
        self.record_fps = 30.0
        self.video_writer = None

        try:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 2500)
            sock.setsockopt(zmq.SNDTIMEO, 1500)
            sock.connect(self.address)
            sock.send(b"info")
            info = sock.recv_json()
            sock.close(linger=0)
            ctx.term()
            if info.get("ok"):
                self.available = True
        except Exception:
            self.available = False

        if not self.available:
            print(f"[camera] Camera server not reachable on {self.address}; skipping camera feed.")
            return

        self.preview_server = viser.ViserServer(port=preview_port, label="Robot Camera")
        self.preview_server.gui.add_markdown("Live robot camera. Press **Ctrl+C in the terminal** to stop replay.")
        print(
            f"[camera] Preview: http://localhost:{self.preview_server.get_port()} "
            "(use the robot computer's hostname when opening remotely).",
            flush=True,
        )
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
        # Wait up to 1.5s for the first frame and publish it to the browser.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and self.latest_frame is None and self.running:
            time.sleep(0.03)
        if self.latest_frame is not None:
            self.update()

    def _fetch_loop(self):
        ctx = zmq.Context()

        def _create_socket():
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.RCVTIMEO, 2500)
            s.setsockopt(zmq.SNDTIMEO, 2500)
            s.connect(self.address)
            return s

        sock = _create_socket()
        try:
            while self.running:
                try:
                    sock.send(b"capture")
                    parts = sock.recv_multipart()
                    if len(parts) >= 2:
                        bgr = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            with self.lock:
                                self.latest_frame = bgr
                                if self.recording and self.record_path is not None:
                                    if self.video_writer is None:
                                        h, w = bgr.shape[:2]
                                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                        self.video_writer = cv2.VideoWriter(
                                            str(self.record_path), fourcc, float(self.record_fps), (w, h)
                                        )
                                    if self.video_writer.isOpened():
                                        self.video_writer.write(bgr)
                except Exception:
                    try:
                        sock.close(linger=0)
                    except Exception:
                        pass
                    sock = _create_socket()
                    time.sleep(0.05)
        finally:
            if sock is not None:
                sock.close(linger=0)
            ctx.term()

    def start_recording(self, output_path: Path, fps: float = 30.0):
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.record_path = output_path
            self.record_fps = fps
            self.recording = True
        print(colored(f"[camera] Started video recording to: {output_path}", "cyan"), flush=True)

    def stop_recording(self):
        writer = None
        saved_path = None
        with self.lock:
            if self.recording:
                self.recording = False
                writer = self.video_writer
                self.video_writer = None
                saved_path = self.record_path
                self.record_path = None
        if writer is not None:
            writer.release()
        if saved_path is not None:
            print(colored(f"[camera] Saved video recording to: {saved_path}", "cyan"), flush=True)

    def update(self):
        if not self.available or not self.running or self.preview_server is None:
            return
        now = time.monotonic()
        if now - self._preview_updated_at < 1.0 / 30.0:
            return
        with self.lock:
            frame = self.latest_frame
        if frame is None or frame is self._preview_frame:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.preview_image is None:
            self.preview_image = self.preview_server.gui.add_image(
                rgb, label="Robot Camera", format="jpeg", jpeg_quality=80
            )
        else:
            self.preview_image.image = rgb
        self._preview_frame = frame
        self._preview_updated_at = now

    def close(self):
        self.stop_recording()
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=4.0)
        if self.preview_server is not None:
            self.preview_server.stop()
            self.preview_server = None


def prompt_with_camera(prompt_text: str, camera_feed=None) -> str:
    print(prompt_text, end="", flush=True)
    while True:
        if camera_feed is not None:
            camera_feed.update()
        r, _, _ = select.select([sys.stdin], [], [], 0.03)
        if r:
            line = sys.stdin.readline()
            return line.strip()


def sleep_with_camera(duration, camera_feed=None, interval=0.03):
    end_time = time.monotonic() + duration
    while time.monotonic() < end_time:
        if camera_feed is not None:
            camera_feed.update()
        remaining = end_time - time.monotonic()
        time.sleep(min(interval, max(0.0, remaining)))

