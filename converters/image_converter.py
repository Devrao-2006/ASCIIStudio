import os
import sys
import cv2

# Ensure app root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ascii_engine import ASCIIEngine
from core.params import ASCIIParams
from core.utils import ensure_file_exists

from typing import Optional

def convert_image(input_path: str, output_path: str, params: Optional[ASCIIParams] = None, engine: Optional[ASCIIEngine] = None) -> None:
    """
    Converts a single image file to an ASCII representation.
    
    Args:
        input_path: Path to the source image.
        output_path: Path to save the processed image.
        params: Optional ASCIIParams instance.
        engine: Optional ASCIIEngine instance.
    """
    ensure_file_exists(input_path)
    
    print(f"Loading image from {input_path}...")
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Failed to load image from {input_path}")
        return

    if engine is None:
        engine = ASCIIEngine()
    if params is None:
        params = ASCIIParams()
    
    print("Applying ASCII rendering pipeline...")
    processed = engine.process_frame(frame, params)
    
    cv2.imwrite(output_path, processed)
    print(f"Success! Saved rendered image to {output_path}")