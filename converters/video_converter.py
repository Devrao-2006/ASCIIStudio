import os
import sys
import cv2
import numpy as np

# Ensure app root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ascii_engine import ASCIIEngine
from core.params import ASCIIParams
from core.utils import ensure_file_exists, print_progress_bar

from typing import Callable, Optional

def convert_video(input_path: str, output_path: str, params: Optional[ASCIIParams] = None, engine: Optional[ASCIIEngine] = None, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
    """
    Processes a video file frame-by-frame and exports an ASCII rendered video.
    
    Args:
        input_path: Path to the source video.
        output_path: Path to save the processed video.
        params: ASCIIParams configuration (defaults to new ASCIIParams).
        engine: ASCIIEngine instance (defaults to new instance).
        progress_callback: Optional callback receiving (current_frame, total_frames).
    """
    ensure_file_exists(input_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Engine adjusts dimensions to be multiples of 8
    target_w = (width // 8) * 8
    target_h = (height // 8) * 8
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    
    if engine is None:
        engine = ASCIIEngine()
    if params is None:
        params = ASCIIParams()
    
    print(f"Starting video conversion: {input_path}")
    print(f"Resolution: {target_w}x{target_h} @ {fps} FPS | Frames: {total_frames}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = engine.process_frame(frame, params)
        if processed.ndim == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
        out.write(processed)
        
        frame_count += 1
        if progress_callback:
            progress_callback(frame_count, total_frames)
        elif frame_count % 5 == 0 or frame_count == total_frames:
            print_progress_bar(frame_count, total_frames, prefix='Progress:', suffix='Complete', length=50)

    cap.release()
    out.release()
    print(f"Success! Saved rendered video to {output_path}")