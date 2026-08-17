import asyncio
import cv2
import numpy as np
import threading
import time
import webbrowser
import sys
import os

import uuid
import shutil

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add app/ dir to path so core/converters packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.ascii_engine import ASCIIEngine
from core.params import ASCIIParams
from converters.video_converter import convert_video

app = FastAPI()

# ── Shared state ────────────────────────────────────────────────────────────
params = ASCIIParams()
engine = ASCIIEngine()
latest_frame: bytes = b""
frame_lock = threading.Lock()

# ── Video Recording State ───────────────────────────────────────────────────
recording_lock = threading.Lock()
is_recording: bool = False
video_writer: cv2.VideoWriter = None
rec_start_time: float = 0.0
rec_filename: str = ""

def _get_captures_dir():
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "videos")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

camera_enabled = threading.Event()
camera_enabled.set()
cached_device_index = None

# ── Camera capture thread ────────────────────────────────────────────────────
def camera_loop(device_index: int):
    global latest_frame, video_writer, cached_device_index
    cached_device_index = device_index
    cap = None

    while True:
        if not camera_enabled.is_set():
            if cap is not None:
                cap.release()
                cap = None
            camera_enabled.wait(timeout=0.2)
            continue

        if cap is None:
            # Auto-detect camera if no index given
            for idx in ([cached_device_index] if cached_device_index is not None else range(4)):
                test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if test.isOpened():
                    cap = test
                    cached_device_index = idx
                    break
                test.release()

            if cap is None:
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Webcam Connected", (440, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                ok, buf = cv2.imencode(".jpg", placeholder)
                if ok:
                    with frame_lock:
                        latest_frame = buf.tobytes()
                time.sleep(1.0)
                continue
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        rendered = engine.process_frame(frame, params)

        # Convert grayscale debug views to BGR for JPEG encoding & video writing
        if rendered.ndim == 2:
            rendered = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)

        # Video recording frame write
        with recording_lock:
            if is_recording:
                if video_writer is None:
                    h_r, w_r = rendered.shape[:2]
                    filepath = os.path.join(_get_captures_dir(), rec_filename)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (w_r, h_r))
                if video_writer is not None:
                    video_writer.write(rendered)
            elif video_writer is not None:
                video_writer.release()
                video_writer = None

        ok, buf = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with frame_lock:
                latest_frame = buf.tobytes()

    if cap is not None:
        cap.release()


# ── MJPEG stream generator ───────────────────────────────────────────────────
def mjpeg_generator():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield boundary + frame + b"\r\n"
        time.sleep(0.016)   # ~60 fps poll cap


# ── Routes ───────────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Camera State Management API Routes ────────────────────────────────────
@app.post("/api/camera/pause")
async def pause_camera():
    camera_enabled.clear()
    return {"camera": "paused"}


@app.post("/api/camera/resume")
async def resume_camera():
    camera_enabled.set()
    return {"camera": "active"}


# ── Recording API Routes ──────────────────────────────────────────────────
@app.post("/api/record/start")
async def start_record():
    global is_recording, rec_start_time, rec_filename
    with recording_lock:
        if not is_recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            rec_filename = f"asciicam_{timestamp}.mp4"
            is_recording = True
            rec_start_time = time.time()
    return {"status": "recording", "filename": rec_filename}


@app.post("/api/record/stop")
async def stop_record():
    global is_recording, rec_filename
    saved = rec_filename
    with recording_lock:
        is_recording = False
    return {"status": "stopped", "filename": saved}


@app.get("/api/record/status")
async def record_status():
    with recording_lock:
        elapsed = time.time() - rec_start_time if is_recording else 0.0
        return {
            "is_recording": is_recording,
            "elapsed": round(elapsed, 1),
            "filename": rec_filename
        }


# ── Studio Conversion API Routes (Image & Video) ──────────────────────────
video_jobs = {}

@app.post("/api/convert/image")
async def api_convert_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return Response(content=b"Invalid image format", status_code=400)
    
    rendered = engine.process_frame(img, params)
    if rendered.ndim == 2:
        rendered = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)
    
    ok, buf = cv2.imencode(".png", rendered)
    if not ok:
        return Response(content=b"Encoding error", status_code=500)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/api/convert/video")
async def api_convert_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "temp_uploads")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "videos")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    in_ext = os.path.splitext(file.filename)[1] or ".mp4"
    in_path = os.path.join(temp_dir, f"upload_{job_id}{in_ext}")
    out_filename = f"asciivideo_{job_id}.mp4"
    out_path = os.path.join(out_dir, out_filename)
    
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    video_jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "filename": out_filename,
        "filepath": out_path
    }
    
    # Snapshot current active params for this conversion
    job_params = ASCIIParams(
        zoom=params.zoom,
        offset_x=params.offset_x,
        offset_y=params.offset_y,
        kernel_size=params.kernel_size,
        sigma=params.sigma,
        sigma_scale=params.sigma_scale,
        tau=params.tau,
        threshold=params.threshold,
        edge_threshold=params.edge_threshold,
        exposure=params.exposure,
        attenuation=params.attenuation,
        blend_with_base=params.blend_with_base,
        draw_edges=params.draw_edges,
        draw_fill=params.draw_fill,
        invert_luminance=params.invert_luminance,
        view_mode=params.view_mode,
        view_uncompressed=params.view_uncompressed,
        theme=params.theme,
        intensity=params.intensity
    )
    
    def run_job():
        def progress_cb(current, total):
            pct = round((current / max(1, total)) * 100, 1)
            video_jobs[job_id]["progress"] = pct
            video_jobs[job_id]["current_frame"] = current
            video_jobs[job_id]["total_frames"] = total
            
        try:
            convert_video(in_path, out_path, params=job_params, engine=engine, progress_callback=progress_cb)
            video_jobs[job_id]["status"] = "completed"
            video_jobs[job_id]["progress"] = 100.0
        except Exception as e:
            video_jobs[job_id]["status"] = "error"
            video_jobs[job_id]["error"] = str(e)
        finally:
            if os.path.exists(in_path):
                try: os.remove(in_path)
                except: pass
                
    threading.Thread(target=run_job, daemon=True).start()
    return {"job_id": job_id, "status": "processing", "filename": out_filename}


@app.get("/api/convert/video/status/{job_id}")
async def api_video_status(job_id: str):
    if job_id not in video_jobs:
        return {"status": "not_found"}
    return video_jobs[job_id]


@app.get("/api/convert/video/download/{job_id}")
async def api_video_download(job_id: str):
    job = video_jobs.get(job_id)
    if not job or job.get("status") != "completed" or not os.path.exists(job["filepath"]):
        return Response(content="File not ready or not found", status_code=404)
    return FileResponse(job["filepath"], media_type="video/mp4", filename=job["filename"])


@app.websocket("/ws")
async def ws_params(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            # Map incoming JSON keys directly onto params fields
            for key, val in data.items():
                if hasattr(params, key):
                    attr = getattr(ASCIIParams, key, None)
                    field_type = type(getattr(params, key))
                    if field_type == bool:
                        setattr(params, key, bool(val))
                    elif field_type == int:
                        setattr(params, key, int(val))
                    else:
                        setattr(params, key, float(val))
    except WebSocketDisconnect:
        pass


# ── Entry point used by main.py ──────────────────────────────────────────────
def run_server(device_index: int = None, host: str = "127.0.0.1", port: int = 8000):
    t = threading.Thread(target=camera_loop, args=(device_index,), daemon=True)
    t.start()

    # Small delay so the camera initialises before the browser opens
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n[+] ASCII Studio is live at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server.\n")

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        pass
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        camera_enabled.clear()
        print("\nASCII Studio stopped.")
