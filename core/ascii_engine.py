import os
import sys
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Dynamic C++/CUDA extension loader with fallback
HAS_CUDA_EXTENSION = False
tile_voting_ext = None

# Check for compiler before attempting PyTorch JIT compilation to avoid noisy PyTorch stacktraces
has_compiler = (os.name != 'nt') or (shutil.which('cl') is not None)

if has_compiler:
    try:
        from torch.utils.cpp_extension import load
        _engine_dir = os.path.dirname(os.path.abspath(__file__))
        _cuda_dir   = os.path.join(_engine_dir, "..", "cuda")
        tile_voting_ext = load(
            name="tile_voting_ext",
            sources=[
                os.path.join(_cuda_dir, "tile_voting.cpp"),
                os.path.join(_cuda_dir, "tile_voting.cu")
            ],
            verbose=False
        )
        HAS_CUDA_EXTENSION = True
        print("ASCIIEngine: Custom CUDA/C++ tile voting extension loaded successfully.")
    except Exception as e:
        print(f"ASCIIEngine: JIT compilation failed. Using PyTorch fallback: {e}")
else:
    print("ASCIIEngine: MSVC C++ compiler (cl.exe) not found. Running on PyTorch GPU fallback.")


def load_or_generate_luts():
    """
    Loads edgesASCII.png and fillASCII.png from the assets/ folder.
    Falls back to programmatically generating default bitmap LUTs if files are missing.
    """
    _assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
    _edges_path = os.path.join(_assets_dir, "edgesASCII.png")
    _fill_path  = os.path.join(_assets_dir, "fillASCII.png")

    if os.path.exists(_edges_path):
        edges_lut = cv2.imread(_edges_path, cv2.IMREAD_GRAYSCALE)
    else:
        edges_patterns = [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18],
            [0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00],
            [0x01, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0],
            [0x80, 0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03]
        ]
        edges_lut = np.zeros((8, 40), dtype=np.uint8)
        for i, pat in enumerate(edges_patterns):
            for r in range(8):
                row_val = pat[r]
                for c in range(8):
                    if (row_val & (1 << (7 - c))) != 0:
                        edges_lut[r, i * 8 + c] = 255
        cv2.imwrite(_edges_path, edges_lut)

    if os.path.exists(_fill_path):
        fill_lut = cv2.imread(_fill_path, cv2.IMREAD_GRAYSCALE)
    else:
        fill_patterns = [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18],
            [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00],
            [0x00, 0x00, 0x00, 0x7E, 0x7E, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00],
            [0x18, 0x18, 0x18, 0xFF, 0xFF, 0x18, 0x18, 0x18],
            [0x18, 0x99, 0x5A, 0x3C, 0x3C, 0x5A, 0x99, 0x18],
            [0x3C, 0x42, 0x81, 0x81, 0x81, 0x81, 0x42, 0x3C],
            [0x24, 0x24, 0xFF, 0x24, 0xFF, 0x24, 0x24, 0x24],
            [0x3C, 0x42, 0x9D, 0xA5, 0xA5, 0x99, 0x42, 0x3C]
        ]
        fill_lut = np.zeros((8, 80), dtype=np.uint8)
        for i, pat in enumerate(fill_patterns):
            for r in range(8):
                row_val = pat[r]
                for c in range(8):
                    if (row_val & (1 << (7 - c))) != 0:
                        fill_lut[r, i * 8 + c] = 255
        cv2.imwrite(_fill_path, fill_lut)

    return edges_lut.astype(np.float32) / 255.0, fill_lut.astype(np.float32) / 255.0


class ASCIIEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"ASCIIEngine initialized on device: {self.device}")
        
        edges_lut, fill_lut = load_or_generate_luts()
        master_lut = np.hstack([edges_lut, fill_lut])
        self.master_lut = torch.from_numpy(master_lut).float().to(self.device)
        
        self.global_h, self.global_w = 0, 0
        self.ly = None
        self.lx = None
        self.ty = None
        self.tx = None
        
        # Color Palettes configured as (Text Color, Background Color) in BGR format
        self.palettes = {
            0: (torch.tensor([0.0, 1.0, 0.0], device=self.device), torch.tensor([0.0, 0.0, 0.0], device=self.device)),         # Classic Green
            1: (torch.tensor([0.15, 0.65, 1.0], device=self.device), torch.tensor([0.0, 0.05, 0.1], device=self.device)),      # Amber Retro
            2: (torch.tensor([0.5, 0.9, 1.0], device=self.device), torch.tensor([0.05, 0.1, 0.0], device=self.device)),        # Gold/Green
            3: (torch.tensor([1.0, 1.0, 1.0], device=self.device), torch.tensor([0.2, 0.05, 0.1], device=self.device)),         # White/Purple
            4: (torch.tensor([0.38, 0.75, 0.85], device=self.device), torch.tensor([0.06, 0.09, 0.04], device=self.device))    # Warm Gold (from your image)
        }

    def update_grid(self, h, w):
        if h == self.global_h and w == self.global_w:
            return
        self.global_h, self.global_w = h, w
        y_indices, x_indices = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        self.ly = y_indices % 8
        self.lx = x_indices % 8
        self.ty = y_indices // 8
        self.tx = x_indices // 8

    def process_frame(self, frame_np, params):
        h, w = frame_np.shape[:2]
        h_new, w_new = (h // 8) * 8, (w // 8) * 8
        if h_new != h or w_new != w:
            frame_np = frame_np[:h_new, :w_new]
            h, w = h_new, w_new

        if params.zoom != 1.0 or params.offset_x != 0.0 or params.offset_y != 0.0:
            cx, cy = w / 2.0, h / 2.0
            M = np.float32([[params.zoom, 0, (1.0 - params.zoom) * cx - params.offset_x * w * params.zoom],
                            [0, params.zoom, (1.0 - params.zoom) * cy + params.offset_y * h * params.zoom]])
            frame_np = cv2.warpAffine(frame_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        frame_t = torch.from_numpy(frame_np).float().to(self.device) / 255.0
        self.update_grid(h, w)

        gray = frame_t[..., 2] * 0.2127 + frame_t[..., 1] * 0.7152 + frame_t[..., 0] * 0.0722
        gray_unsqueeze = gray.unsqueeze(0).unsqueeze(0)
        
        k_size = params.kernel_size * 2 + 1
        sigma1 = max(0.1, params.sigma)
        sigma2 = max(0.1, params.sigma * params.sigma_scale)
        
        blur1 = TF.gaussian_blur(gray_unsqueeze, [k_size, k_size], [sigma1, sigma1])
        blur2 = TF.gaussian_blur(gray_unsqueeze, [k_size, k_size], [sigma2, sigma2])
        
        diff = blur1 - params.tau * blur2
        edges = (diff >= params.threshold).float().squeeze()

        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], device=self.device).view(1,1,3,3)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], device=self.device).view(1,1,3,3)
        
        edges_unsqueeze = edges.unsqueeze(0).unsqueeze(0)
        Gx = F.conv2d(edges_unsqueeze, scharr_x, padding=1).squeeze()
        Gy = F.conv2d(edges_unsqueeze, scharr_y, padding=1).squeeze()

        magnitude = torch.sqrt(Gx * Gx + Gy * Gy)
        has_gradient = magnitude > 1e-5

        theta = torch.atan2(Gy, Gx)
        abs_theta = torch.abs(theta) / torch.pi

        directions = torch.full_like(theta, -1, dtype=torch.long)
        
        v_mask = ((abs_theta >= 0.0) & (abs_theta < 0.05)) | ((abs_theta > 0.9) & (abs_theta <= 1.0))
        h_mask = (abs_theta > 0.45) & (abs_theta < 0.55)
        d1_mask = (abs_theta >= 0.05) & (abs_theta <= 0.45)
        d2_mask = (abs_theta >= 0.55) & (abs_theta <= 0.9)
        
        pos_mask = theta > 0

        directions[v_mask] = 0
        directions[h_mask] = 1
        directions[d1_mask & pos_mask] = 2
        directions[d1_mask & ~pos_mask] = 3
        directions[d2_mask & pos_mask] = 3
        directions[d2_mask & ~pos_mask] = 2

        directions[~has_gradient] = -1

        if HAS_CUDA_EXTENSION:
            dominant_dirs = tile_voting_ext.tile_voting(directions, params.edge_threshold)
        else:
            tiles = directions.view(h // 8, 8, w // 8, 8).permute(0, 2, 1, 3).contiguous().view(h // 8, w // 8, 64)
            counts = torch.stack([
                (tiles == 0).sum(dim=-1),
                (tiles == 1).sum(dim=-1),
                (tiles == 2).sum(dim=-1),
                (tiles == 3).sum(dim=-1)
            ], dim=-1)
            max_counts, dominant_dirs = torch.max(counts, dim=-1)
            dominant_dirs[max_counts < params.edge_threshold] = -1


        downscale_lum = F.interpolate(gray_unsqueeze, size=(h//8, w//8), mode='area').squeeze()
        frame_permuted = frame_t.permute(2, 0, 1).unsqueeze(0)
        downscale_color = F.interpolate(frame_permuted, size=(h//8, w//8), mode='area').squeeze().permute(1, 2, 0)

        lum_val = torch.clamp(torch.pow(downscale_lum * params.exposure, params.attenuation), 0.0, 1.0)
        if params.invert_luminance:
            lum_val = 1.0 - lum_val
        char_idx = torch.clamp(torch.floor(lum_val * 10).long(), 0, 9)

        tile_d = directions if params.view_uncompressed else dominant_dirs[self.ty, self.tx]
        tile_char = char_idx[self.ty, self.tx]

        is_edge = (tile_d >= 0) & params.draw_edges
        
        lut_x = torch.zeros((h, w), dtype=torch.long, device=self.device)
        lut_x[is_edge] = self.lx[is_edge] + (tile_d[is_edge] + 1) * 8
        
        fill_mask = ~is_edge
        if params.draw_fill:
            lut_x[fill_mask] = self.lx[fill_mask] + tile_char[fill_mask] * 8 + 40
        else:
            lut_x[fill_mask] = 0

        ascii_val = self.master_lut[self.ly, lut_x]

        tile_color = downscale_color[self.ty, self.tx]
        ascii_color, bg_color = self.palettes.get(params.theme, self.palettes[0])
        
        # Apply intensity multiplier to theme ascii_color so original camera colors aren't blown out to white
        ascii_color_boosted = ascii_color * params.intensity
        blend = params.blend_with_base
        text_color = blend * tile_color + (1.0 - blend) * ascii_color_boosted

        # Ambient cell background tinting when blend > 0 to preserve scene context
        bg_color_tinted = (1.0 - blend) * bg_color + blend * (tile_color * 0.18)

        ascii_val_exp = ascii_val.unsqueeze(-1)
        output_color = (1.0 - ascii_val_exp) * bg_color_tinted + ascii_val_exp * text_color
        
        output_color_u8 = (torch.clamp(output_color, 0.0, 1.0) * 255.0).byte().cpu().numpy()

        if params.view_mode == 0:
            return output_color_u8
        elif params.view_mode == 1:
            return (edges * 255.0).byte().cpu().numpy()
        elif params.view_mode == 4:
            return (gray * 255.0).byte().cpu().numpy()
        else:
            return output_color_u8