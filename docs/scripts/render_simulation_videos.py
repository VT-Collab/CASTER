#!/usr/bin/env python3
"""Render all browser simulation rollouts from their calibrated C2R views."""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDER_PAGE = ROOT / 'scripts' / 'simulation_video_renderer.html'
SAFE_NAME = re.compile(r'^[a-z0-9-]+$')
SAFE_FRAME = re.compile(r'^frame-[0-9]{5}\.jpg$')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path.home() / 'Downloads' / 'CASTER_simulation_rollouts')
    parser.add_argument('--chrome', type=Path,
                        default=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'))
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--only', action='append', metavar='TASK_ID',
                        help='Render only this task ID; repeat for multiple tasks')
    args = parser.parse_args()
    if not args.chrome.is_file():
        parser.error(f'Chrome not found: {args.chrome}')
    if not shutil.which(args.ffmpeg):
        parser.error(f'ffmpeg not found: {args.ffmpeg}')

    examples = json.loads((ROOT / 'static/resources/pointcloud/manifest.json').read_text())['examples']
    if args.only:
        requested = set(args.only)
        examples = [entry for entry in examples if entry['id'] in requested]
        found = {entry['id'] for entry in examples}
        if found != requested:
            parser.error('Unknown task ID(s): ' + ', '.join(sorted(requested - found)))
    identifiers = [entry['id'] for entry in examples]
    done = threading.Event()
    failure = []

    with tempfile.TemporaryDirectory(prefix='caster-video-render-') as temp_name:
        temp = Path(temp_name)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *handler_args, **handler_kwargs):
                super().__init__(*handler_args, directory=str(ROOT), **handler_kwargs)

            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path.split('?', 1)[0] == '/__render/page':
                    data = RENDER_PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                super().do_GET()

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                parts = self.path.strip('/').split('/')
                try:
                    if len(parts) == 4 and parts[:2] == ['__render', 'frame']:
                        scene, frame = parts[2:]
                        if scene not in identifiers or not SAFE_NAME.fullmatch(scene) or not SAFE_FRAME.fullmatch(frame):
                            raise ValueError('Invalid frame path')
                        folder = temp / scene
                        folder.mkdir(exist_ok=True)
                        (folder / frame).write_bytes(body)
                    elif len(parts) == 3 and parts[:2] == ['__render', 'scene-done']:
                        scene = parts[2]
                        if scene not in identifiers:
                            raise ValueError('Invalid scene')
                        count = len(list((temp / scene).glob('frame-*.jpg')))
                        print(f'captured {scene}: {count} frames', flush=True)
                    elif parts == ['__render', 'all-done']:
                        done.set()
                    elif parts == ['__render', 'error']:
                        failure.append(body.decode('utf-8', errors='replace'))
                        done.set()
                    else:
                        raise ValueError('Unknown render endpoint')
                except Exception as error:
                    self.send_error(400, str(error))
                    return
                self.send_response(204)
                self.end_headers()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        profile = temp / 'chrome-profile'
        only_query = ','.join(identifiers)
        command = [
            str(args.chrome), '--headless=new', '--enable-unsafe-swiftshader',
            '--ignore-gpu-blocklist', '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding', f'--user-data-dir={profile}',
            '--window-size=900,540',
            f'http://127.0.0.1:{server.server_port}/__render/page?only={only_query}',
        ]
        chrome = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            if not done.wait(args.timeout):
                raise TimeoutError(f'Rendering did not finish within {args.timeout} seconds')
            if failure:
                raise RuntimeError('Browser renderer failed:\n' + failure[0])
            args.output.mkdir(parents=True, exist_ok=True)
            for identifier in identifiers:
                frames = temp / identifier
                if not frames.is_dir():
                    raise RuntimeError(f'No frames captured for {identifier}')
                destination = args.output / f'{identifier}.mp4'
                subprocess.run([
                    args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    '-framerate', '30', '-i', str(frames / 'frame-%05d.jpg'),
                    '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination),
                ], check=True)
                print(f'wrote {destination}', flush=True)
        finally:
            chrome.terminate()
            try:
                chrome.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome.kill()
            server.shutdown()
            server.server_close()
        if chrome.returncode not in (0, -15) and not failure:
            message = chrome.stderr.read() if chrome.stderr else ''
            raise RuntimeError(f'Chrome exited with status {chrome.returncode}: {message[-2000:]}')


if __name__ == '__main__':
    main()
