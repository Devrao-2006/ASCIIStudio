import cv2
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.params import ASCIIParams

def nothing(x):
    pass

def setup_ui(control_window: str) -> None:
    """
    Initializes the OpenCV trackbars for the settings menu.
    
    Args:
        control_window: The name of the OpenCV window to attach trackbars to.
    """
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 450, 600)

    cv2.createTrackbar("Zoom x10", control_window, 10, 50, nothing)       
    cv2.createTrackbar("Offset X x100", control_window, 50, 100, nothing) 
    cv2.createTrackbar("Offset Y x100", control_window, 50, 100, nothing)
    
    cv2.createTrackbar("Kernel Size", control_window, 2, 10, nothing)     
    cv2.createTrackbar("Sigma x10", control_window, 20, 50, nothing)      
    cv2.createTrackbar("Sigma Scale x10", control_window, 16, 50, nothing) 
    cv2.createTrackbar("Tau x100", control_window, 100, 110, nothing)     
    cv2.createTrackbar("Threshold x1000", control_window, 5, 100, nothing) 
    cv2.createTrackbar("Edge Thresh", control_window, 8, 64, nothing)     
    
    cv2.createTrackbar("Exposure x10", control_window, 10, 50, nothing)   
    cv2.createTrackbar("Attenuation x10", control_window, 10, 50, nothing) 
    cv2.createTrackbar("Blend Base x10", control_window, 10, 10, nothing)  
    
    cv2.createTrackbar("Draw Edges", control_window, 1, 1, nothing)
    cv2.createTrackbar("Draw Fill", control_window, 1, 1, nothing)
    cv2.createTrackbar("Invert ASCII", control_window, 0, 1, nothing)

    cv2.createTrackbar("View Mode", control_window, 0, 4, nothing)

def read_ui_params(control_window: str, params: ASCIIParams) -> None:
    """
    Reads the current positions of all trackbars and updates the params object.
    
    Args:
        control_window: The name of the window containing the trackbars.
        params: The ASCIIParams dataclass instance to update.
    """
    try:
        zoom = cv2.getTrackbarPos("Zoom x10", control_window) / 10.0
        params.zoom = max(0.1, zoom)
        
        params.offset_x = (cv2.getTrackbarPos("Offset X x100", control_window) - 50) / 50.0
        params.offset_y = (cv2.getTrackbarPos("Offset Y x100", control_window) - 50) / 50.0

        params.kernel_size = max(1, cv2.getTrackbarPos("Kernel Size", control_window))
        params.sigma = cv2.getTrackbarPos("Sigma x10", control_window) / 10.0
        params.sigma_scale = cv2.getTrackbarPos("Sigma Scale x10", control_window) / 10.0
        params.tau = cv2.getTrackbarPos("Tau x100", control_window) / 100.0
        params.threshold = cv2.getTrackbarPos("Threshold x1000", control_window) / 1000.0
        params.edge_threshold = cv2.getTrackbarPos("Edge Thresh", control_window)

        params.exposure = cv2.getTrackbarPos("Exposure x10", control_window) / 10.0
        params.attenuation = cv2.getTrackbarPos("Attenuation x10", control_window) / 10.0
        params.blend_with_base = cv2.getTrackbarPos("Blend Base x10", control_window) / 10.0
        
        params.draw_edges = cv2.getTrackbarPos("Draw Edges", control_window) == 1
        params.draw_fill = cv2.getTrackbarPos("Draw Fill", control_window) == 1
        params.invert_luminance = cv2.getTrackbarPos("Invert ASCII", control_window) == 1
        
        view_mode = cv2.getTrackbarPos("View Mode", control_window)
        params.view_mode = view_mode
        params.view_uncompressed = (view_mode == 3)
    except cv2.error:
        # Window was closed, ignore parameter reading
        pass