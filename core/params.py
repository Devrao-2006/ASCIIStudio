from dataclasses import dataclass

@dataclass
class ASCIIParams:
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    
    kernel_size: int = 2
    sigma: float = 2.0
    sigma_scale: float = 1.6
    tau: float = 1.0
    threshold: float = 0.005
    edge_threshold: int = 8
    
    exposure: float = 1.8
    attenuation: float = 0.85
    blend_with_base: float = 1.0
    
    draw_edges: bool = True
    draw_fill: bool = True
    invert_luminance: bool = False
    
    view_mode: int = 0
    view_uncompressed: bool = False
    
    # New palette and intensity parameters
    theme: int = 0
    intensity: float = 1.0