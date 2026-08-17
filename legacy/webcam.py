import cv2
import os
import datetime
import numpy as np
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.ascii_engine import ASCIIEngine
from core.params import ASCIIParams

def nothing(x):
    pass

def _get_timestamp():
    """Returns a clean timestamp string for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_output_dir(folder: str) -> str:
    """Creates the output directory if it doesn't exist and returns the path."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def _draw_hud(frame: np.ndarray, is_recording: bool, rec_frame_count: int, fps: float) -> np.ndarray:
    """
    Draws a transparent heads-up display (HUD) overlay onto the output frame.
    Shows FPS, recording indicator, and keyboard shortcut hints.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # --- Bottom bar background ---
    bar_h = 36
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # --- FPS counter (bottom-left) ---
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # --- Keyboard hints (bottom-center) ---
    hints = "  [P] Photo    [R] Record    [ESC] Quit"
    (tw, th), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hints, (w // 2 - tw // 2, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    # --- Recording indicator (top-right) ---
    if is_recording:
        # Blinking red dot (blinks every 20 frames)
        if (rec_frame_count // 20) % 2 == 0:
            cv2.circle(frame, (w - 20, 18), 8, (0, 0, 220), -1)
        rec_label = f"REC  {rec_frame_count // 30:02d}s"
        cv2.putText(frame, rec_label, (w - 110, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "LIVE", (w - 55, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 1, cv2.LINE_AA)

    return frame

def _flash_capture_indicator(frame: np.ndarray) -> np.ndarray:
    """Briefly brightens the frame to give a camera-shutter flash effect."""
    bright = np.clip(frame.astype(np.int32) + 80, 0, 255).astype(np.uint8)
    return bright

def run_webcam(device_index=None):
    engine = ASCIIEngine()

    # ------------------------------------------------------------------
    # Camera initialisation
    # ------------------------------------------------------------------
    cap = None
    if device_index is not None:
        # Try the user-specified device directly
        candidate = cv2.VideoCapture(device_index)
        if candidate.isOpened():
            cap = candidate
            print(f"Camera opened on user-specified device index {device_index}.")
        else:
            candidate.release()
            print(f"Warning: Could not open user-specified camera device index {device_index}.")

    if cap is None:
        # Fall back to auto-detecting
        print("Searching for available camera devices...")
        for dev_idx in [0, 1, 2, 3]:
            candidate = cv2.VideoCapture(dev_idx)
            if candidate.isOpened():
                cap = candidate
                print(f"Camera opened on auto-detected device index {dev_idx}.")
                break
            candidate.release()

    if cap is None:
        print("Error: Could not find any open camera device.")
        return

    # Read actual camera resolution (do NOT force-override - keeps aspect ratio)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera resolution: {cam_w}x{cam_h} @ {cam_fps:.1f} FPS")

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    photos_dir = _ensure_output_dir("captures/photos")
    videos_dir = _ensure_output_dir("captures/videos")

    # ------------------------------------------------------------------
    # Window setup - two windows: controls + output
    # ------------------------------------------------------------------
    control_window = "Parameters"
    output_window  = "ASCII Camera"

    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 450, 700)

    cv2.namedWindow(output_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(output_window, min(cam_w, 1280), min(cam_h, 720))
    cv2.setWindowProperty(output_window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # ------------------------------------------------------------------
    # Trackbar definitions
    # ------------------------------------------------------------------
    cv2.createTrackbar("Zoom x10",         control_window, 10,  50,  nothing)
    cv2.createTrackbar("Offset X x100",    control_window, 50,  100, nothing)
    cv2.createTrackbar("Offset Y x100",    control_window, 50,  100, nothing)
    cv2.createTrackbar("Kernel Size",      control_window, 2,   10,  nothing)
    cv2.createTrackbar("Sigma x10",        control_window, 20,  50,  nothing)
    cv2.createTrackbar("Sigma Scale x10",  control_window, 16,  50,  nothing)
    cv2.createTrackbar("Tau x100",         control_window, 100, 110, nothing)
    cv2.createTrackbar("Threshold x1000",  control_window, 5,   100, nothing)
    cv2.createTrackbar("Edge Thresh",      control_window, 8,   64,  nothing)
    cv2.createTrackbar("Exposure x10",     control_window, 10,  50,  nothing)
    cv2.createTrackbar("Attenuation x10",  control_window, 10,  50,  nothing)
    cv2.createTrackbar("Blend Base x10",   control_window, 10,  10,  nothing)
    cv2.createTrackbar("Draw Edges",       control_window, 1,   1,   nothing)
    cv2.createTrackbar("Draw Fill",        control_window, 1,   1,   nothing)
    cv2.createTrackbar("Invert ASCII",     control_window, 0,   1,   nothing)
    cv2.createTrackbar("View Mode",        control_window, 0,   4,   nothing)
    cv2.createTrackbar("Theme",            control_window, 0,   4,   nothing)
    cv2.createTrackbar("Intensity x10",    control_window, 20,  50,  nothing)

    print("ASCII Camera ready.")
    print("  [P]   – Capture photo  (saved to captures/photos/)")
    print("  [R]   – Start/stop recording video  (saved to captures/videos/)")
    print("  [ESC] – Quit")

    # ------------------------------------------------------------------
    # State variables
    # ------------------------------------------------------------------
    fps_history = []
    frame_count = 0

    is_recording   = False
    video_writer   = None
    rec_frame_count = 0
    flash_frames   = 0          # counts down to 0 after a photo flash

    while True:
        # ---- Window-closed guard ----
        try:
            if cv2.getWindowProperty(output_window,  cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(control_window, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        # ---- Capture frame ----
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            continue

        # ---- Read trackbars ----
        zoom        = max(0.1, cv2.getTrackbarPos("Zoom x10",        control_window) / 10.0)
        offset_x    = (cv2.getTrackbarPos("Offset X x100",   control_window) - 50) / 50.0
        offset_y    = (cv2.getTrackbarPos("Offset Y x100",   control_window) - 50) / 50.0
        kernel_size = max(1, cv2.getTrackbarPos("Kernel Size",     control_window))
        sigma       = cv2.getTrackbarPos("Sigma x10",        control_window) / 10.0
        sigma_scale = cv2.getTrackbarPos("Sigma Scale x10",  control_window) / 10.0
        tau         = cv2.getTrackbarPos("Tau x100",         control_window) / 100.0
        threshold   = cv2.getTrackbarPos("Threshold x1000",  control_window) / 1000.0
        edge_thresh = cv2.getTrackbarPos("Edge Thresh",      control_window)
        exposure    = cv2.getTrackbarPos("Exposure x10",     control_window) / 10.0
        attenuation = cv2.getTrackbarPos("Attenuation x10",  control_window) / 10.0
        blend       = cv2.getTrackbarPos("Blend Base x10",   control_window) / 10.0
        draw_edges  = cv2.getTrackbarPos("Draw Edges",       control_window) == 1
        draw_fill   = cv2.getTrackbarPos("Draw Fill",        control_window) == 1
        invert      = cv2.getTrackbarPos("Invert ASCII",     control_window) == 1
        view_mode   = cv2.getTrackbarPos("View Mode",        control_window)
        theme       = cv2.getTrackbarPos("Theme",            control_window)
        intensity   = cv2.getTrackbarPos("Intensity x10",    control_window) / 10.0

        params = ASCIIParams(
            zoom=zoom, offset_x=offset_x, offset_y=offset_y,
            kernel_size=kernel_size, sigma=sigma, sigma_scale=sigma_scale,
            tau=tau, threshold=threshold, edge_threshold=edge_thresh,
            exposure=exposure, attenuation=attenuation, blend_with_base=blend,
            draw_edges=draw_edges, draw_fill=draw_fill, invert_luminance=invert,
            view_mode=view_mode, view_uncompressed=(view_mode == 3),
            theme=theme, intensity=intensity
        )

        # ---- Process frame through ASCII engine ----
        start_tick = cv2.getTickCount()
        processed = engine.process_frame(raw_frame, params)
        end_tick   = cv2.getTickCount()

        # ---- FPS calculation ----
        elapsed = (end_tick - start_tick) / cv2.getTickFrequency()
        if elapsed > 0:
            fps_history.append(1.0 / elapsed)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

        # ---- Video recording: feed processed frame to writer ----
        if is_recording and video_writer is not None:
            # Ensure frame is BGR uint8 (required by VideoWriter)
            if processed.ndim == 2:
                write_frame = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                write_frame = processed
            video_writer.write(write_frame)
            rec_frame_count += 1

        # ---- HUD overlay ----
        if processed.ndim == 2:
            # Convert grayscale debug views to BGR before drawing HUD text
            display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            display = processed.copy()

        if flash_frames > 0:
            display = _flash_capture_indicator(display)
            flash_frames -= 1

        display = _draw_hud(display, is_recording, rec_frame_count, avg_fps)

        cv2.imshow(output_window, display)
        cv2.setWindowTitle(output_window, f"ASCII Camera | FPS: {int(avg_fps)}")

        # ---- Key handling ----
        key = cv2.waitKey(1) & 0xFF

        # ESC – quit
        if key == 27:
            break

        # P – photo capture (saves the filtered frame as a PNG)
        elif key == ord('p') or key == ord('P'):
            fname = os.path.join(photos_dir, f"photo_{_get_timestamp()}.png")
            save_frame = processed if processed.ndim == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(fname, save_frame)
            flash_frames = 6           # trigger flash indicator for ~6 display frames
            print(f"[PHOTO] Saved → {fname}")

        # R – toggle video recording
        elif key == ord('r') or key == ord('R'):
            if not is_recording:
                # Start recording
                h_out, w_out = processed.shape[:2]
                vname = os.path.join(videos_dir, f"video_{_get_timestamp()}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(vname, fourcc, cam_fps, (w_out, h_out))
                is_recording    = True
                rec_frame_count = 0
                print(f"[REC] Recording started → {vname}")
            else:
                # Stop recording
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                is_recording = False
                print(f"[REC] Recording stopped. Frames saved: {rec_frame_count}")
                rec_frame_count = 0

        frame_count += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if is_recording and video_writer is not None:
        video_writer.release()
        print("[REC] Recording auto-saved on exit.")

    cap.release()
    cv2.destroyAllWindows()
    print("ASCII Camera terminated cleanly.")