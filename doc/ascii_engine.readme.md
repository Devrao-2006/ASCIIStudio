# ASCIIEngine

A GPU-accelerated image-to-ASCII renderer that converts camera/image frames into an **8×8 tile-based ASCII-style visualization**. The engine combines multi-scale edge detection, directional edge classification, luminance quantization, bitmap glyph lookup tables, and color blending to produce a stylized ASCII representation while preserving scene structure and color.

## Overview

The processing pipeline can be summarized as:

```text
Input Image
    │
    ├── Zoom / Offset
    │
    ├── Luminance
    │      │
    │      ├── Gaussian Blur × 2
    │      ├── Difference-of-Gaussians
    │      └── Threshold → Edge Map
    │
    ├── Scharr Gradient
    │      │
    │      └── 4-Way Edge Direction
    │
    ├── 8×8 Tile Analysis
    │      ├── Dominant Edge Direction
    │      └── Average Luminance
    │
    ├── Glyph Selection
    │      ├── Directional Edge Glyph
    │      └── Brightness Fill Glyph
    │
    ├── Tile Color Extraction
    │
    └── Theme / Color Blending
           │
           ▼
      ASCII-Style Output
```

---

## 1. Tile-Based Rendering

The image is processed in **8×8 pixel cells**.

Input dimensions are cropped to the nearest multiple of 8:

```python
h_new = (h // 8) * 8
w_new = (w // 8) * 8
```

Each 8×8 image region corresponds to one logical ASCII cell.

For example:

```text
1920 × 1080 input
        ↓
1920 × 1072 processing area
        ↓
240 × 134 ASCII cells
```

The final output remains a full-resolution image; the ASCII characters are represented as 8×8 bitmap glyphs rather than as a text string.

---

## 2. Image Transformation

Before analysis, optional zoom and positional offsets can be applied.

The transformation supports:

* Zooming around the image center
* Horizontal offset
* Vertical offset

The transformed frame is resampled using OpenCV's `warpAffine`.

---

## 3. Luminance Extraction

The input BGR image is converted to perceptual luminance using:

```text
L = 0.2127R + 0.7152G + 0.0722B
```

This produces a normalized grayscale image in the range:

```text
0.0 → black
1.0 → white
```

The luminance image is used for both edge detection and ASCII character selection.

---

# Edge Detection

## 4. Multi-Scale Edge Detection

Instead of directly using Canny or Sobel edge detection, ASCIIEngine uses a **Difference-of-Gaussians-style filter**.

Two Gaussian-blurred versions of the luminance image are generated:

```text
Blur₁ = Gaussian(L, σ)
Blur₂ = Gaussian(L, σ × sigma_scale)
```

They are combined as:

```text
D = Blur₁ - τ × Blur₂
```

where:

* `σ` = base blur radius
* `sigma_scale` = scale multiplier for the second blur
* `τ` = relative weighting
* `D` = multi-scale response

The response is then thresholded:

```text
Edge(x,y) = D(x,y) >= threshold
```

This creates a binary edge map.

This approach emphasizes brightness structures at a particular spatial scale rather than treating every high-frequency change as an edge.

---

# Edge Direction

## 5. Scharr Gradient

The binary edge map is passed through horizontal and vertical Scharr filters.

### X Gradient

```text
-3   0   3
-10  0  10
-3   0   3
```

### Y Gradient

```text
-3  -10  -3
 0    0   0
 3   10   3
```

This produces:

```text
Gx = horizontal gradient
Gy = vertical gradient
```

Gradient magnitude is calculated as:

```text
Magnitude = √(Gx² + Gy²)
```

Pixels with negligible magnitude are treated as having no valid edge direction.

---

## 6. Four-Way Direction Quantization

The gradient orientation is calculated using:

```text
θ = atan2(Gy, Gx)
```

The continuous angle is then quantized into four directional classes:

```text
0 → horizontal
1 → vertical
2 → diagonal
3 → opposite diagonal
```

Pixels without a meaningful gradient are assigned:

```text
-1
```

The four directions allow the renderer to select different directional ASCII glyphs depending on the shape of an edge.

---

# Tile Analysis

## 7. Dominant Edge Direction

Each 8×8 tile contains up to 64 directional pixels.

For every tile, ASCIIEngine counts how many pixels belong to each of the four directions:

```text
direction 0 → count
direction 1 → count
direction 2 → count
direction 3 → count
```

The direction with the highest count becomes the tile's dominant direction.

For example:

```text
Direction 0:  5
Direction 1:  3
Direction 2: 42  ← dominant
Direction 3:  8
```

The tile is therefore classified as:

```text
dominant_direction = 2
```

A minimum `edge_threshold` can be used to reject weak or ambiguous tiles.

If no direction has enough supporting pixels:

```text
dominant_direction = -1
```

---

## 8. CUDA Acceleration

When available, dominant-direction calculation is delegated to the custom CUDA/C++ extension:

```python
tile_voting_ext.tile_voting(...)
```

If compilation or loading fails, the engine automatically falls back to a PyTorch implementation.

This provides two execution paths:

```text
CUDA extension available
        ↓
Custom C++/CUDA tile voting
```

or:

```text
CUDA extension unavailable
        ↓
PyTorch tile voting
```

The fallback preserves functionality without requiring the custom extension to compile successfully.

---

# Brightness-Based Glyph Selection

## 9. Average Luminance Per Tile

The luminance image is downscaled from:

```text
H × W
```

to:

```text
H/8 × W/8
```

using area interpolation.

Each output pixel therefore represents the average luminance of one 8×8 cell.

Conceptually:

```text
8×8 pixels
    ↓
average luminance
    ↓
one ASCII brightness value
```

---

## 10. Exposure and Attenuation

Tile luminance is adjusted using:

```text
L' = clamp((L × exposure)^attenuation, 0, 1)
```

This allows the brightness response to be modified independently of the original image.

Optional luminance inversion is then applied:

```text
L' = 1 - L'
```

---

## 11. Ten Brightness Levels

The adjusted luminance is quantized into ten character classes:

```text
character_index = floor(L' × 10)
```

and clamped to:

```text
0 → 9
```

Therefore:

```text
0.00 – 0.09 → glyph 0
0.10 – 0.19 → glyph 1
0.20 – 0.29 → glyph 2
...
0.90 – 1.00 → glyph 9
```

These ten glyphs provide progressively different fill densities.

---

# Glyph Lookup Tables

## 12. Bitmap LUT Architecture

ASCIIEngine does not construct glyphs procedurally during every frame.

Instead, it uses two bitmap lookup tables:

```text
edgesASCII.png
fillASCII.png
```

Each glyph is represented by an **8×8 bitmap**.

If the files are missing, default LUTs are generated programmatically.

The LUTs are combined into a single master table:

```text
master_lut = [edge glyphs | fill glyphs]
```

The resulting layout is:

```text
Edge Glyphs
────────────────────────
0–7    base/empty
8–15   direction 0
16–23  direction 1
24–31  direction 2
32–39  direction 3

Fill Glyphs
────────────────────────
40–47    brightness 0
48–55    brightness 1
56–63    brightness 2
...
112–119  brightness 9
```

Each glyph occupies eight columns and eight rows.

---

# Glyph Selection

## 13. Edge vs. Fill Rendering

For every pixel, the engine determines which 8×8 tile it belongs to and retrieves:

* The tile's dominant edge direction
* The tile's brightness character index

The renderer then chooses between an edge glyph and a fill glyph.

Conceptually:

```text
if valid_edge and draw_edges:
    use directional edge glyph
else:
    use brightness fill glyph
```

This produces two fundamentally different visual representations:

```text
Strong structure
      ↓
directional ASCII glyph
```

and:

```text
No dominant structure
      ↓
brightness/density glyph
```

---

## 14. Uncompressed Edge Mode

Normally, edge direction is calculated once per 8×8 tile.

When `view_uncompressed` is enabled, the renderer instead uses the original per-pixel direction map.

This allows the output to preserve finer edge-direction information rather than forcing the entire 8×8 cell to use one dominant direction.

---

# Color Processing

## 15. Average Tile Color

In parallel with luminance processing, the original RGB/BGR image is downscaled to the ASCII grid resolution.

Each 8×8 cell therefore receives an average scene color:

```text
Tile
 ↓
average color
 ↓
[R, G, B]
```

This allows the ASCII representation to retain some of the original image's color information.

---

## 16. Color Themes

ASCIIEngine provides several predefined color themes.

Each theme contains:

```text
ASCII/Text Color
Background Color
```

Examples include:

```text
Classic Green
Amber Retro
Gold/Green
White/Purple
Warm Gold
```

The theme color is used as the primary ASCII color when scene-color blending is disabled.

---

## 17. ASCII Intensity

The theme's text color is multiplied by:

```text
intensity
```

Conceptually:

```text
text_color = theme_color × intensity
```

This controls the overall brightness of the ASCII glyphs.

---

## 18. Original-Color Blending

The final text color can be blended between the selected theme and the original image:

```text
text_color =
    blend × tile_color
    +
    (1 - blend) × theme_color
```

Therefore:

```text
blend = 0
    → entirely theme-colored ASCII

blend = 1
    → entirely original scene-colored ASCII
```

Intermediate values produce a combination of both.

---

## 19. Background Tinting

The background is also slightly influenced by the original tile color:

```text
background =
    (1 - blend) × theme_background
    +
    blend × tile_color × 0.18
```

The `0.18` multiplier keeps the scene tint relatively subtle, preventing the background from becoming as bright as the ASCII glyph itself.

---

# Final Composition

## 20. Bitmap Glyph Compositing

The LUT produces an 8×8 glyph mask:

```text
ascii_val ∈ [0,1]
```

The final pixel color is calculated as:

```text
Output =
    (1 - glyph) × Background
    +
    glyph × Text
```

Therefore:

```text
glyph = 0
    → background

glyph = 1
    → ASCII/text color
```

This effectively renders the selected 8×8 bitmap glyph into the corresponding image cell.

---

# Output Modes

`process_frame()` supports several viewing modes.

### Normal ASCII mode

```python
view_mode == 0
```

Returns the fully composited color ASCII image.

### Edge visualization

```python
view_mode == 1
```

Returns the raw binary edge map:

```text
0   → black
255 → white
```

This is useful for debugging the edge detector.

### Luminance visualization

```python
view_mode == 4
```

Returns the grayscale luminance image.

Other modes currently fall back to the normal ASCII output.

---

# Performance Architecture

ASCIIEngine automatically selects the available compute device:

```python
cuda if torch.cuda.is_available() else cpu
```

The main image-processing operations are implemented with PyTorch and torchvision, allowing them to execute on the GPU.

The optional custom CUDA extension specifically accelerates tile voting.

The overall architecture is therefore:

```text
                  ASCIIEngine
                      │
             ┌────────┴────────┐
             │                 │
          PyTorch          CUDA Extension
             │                 │
     Image processing    Tile direction voting
             │                 │
             └────────┬────────┘
                      ▼
                 Final Render
```

---

# Processing Summary

For every frame, ASCIIEngine effectively performs:

```text
1. Crop image to 8×8 boundaries
2. Apply zoom/offset
3. Convert image to luminance
4. Generate two Gaussian-blurred images
5. Calculate multi-scale edge response
6. Threshold the response
7. Calculate Scharr gradients
8. Quantize gradients into four directions
9. Analyze each 8×8 tile
10. Determine dominant edge direction
11. Calculate average tile luminance
12. Apply exposure and attenuation
13. Quantize luminance into 10 levels
14. Select either an edge or fill glyph
15. Calculate average tile color
16. Apply the selected color theme
17. Blend theme color with original scene color
18. Composite the 8×8 bitmap glyph
19. Return the rendered frame
```

## Core Design

The renderer can be thought of as combining **structure + brightness + color**:

```text
                  INPUT IMAGE
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Structure    Brightness     Color
          │            │            │
          ▼            ▼            ▼
    Edge direction   0–9 level   Tile average
          │            │            │
          └──────┬─────┴──────┬─────┘
                 │            │
                 ▼            ▼
             Glyph LUT    Color system
                 │            │
                 └──────┬─────┘
                        ▼
                 ASCII FRAME
```

The key design choice is that **edges determine the shape of the glyph, while luminance determines the fill glyph when no strong edge is present**. This lets the renderer preserve recognizable contours while still representing broad light and dark regions.
