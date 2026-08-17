# UI Design Reference — ASCII CAM Web Interface

This document describes the architecture, design decisions, and parameter contracts of the browser-based control interface.

---

## Architecture

```
Browser (index.html + style.css + main.js)
    │
    ├─ GET /            → Serves index.html (FastAPI StaticFiles)
    ├─ GET /static/*    → CSS, JS assets
    ├─ GET /video_feed  → MJPEG stream (multipart/x-mixed-replace)
    └─ WS  /ws          → Real-time parameter updates (JSON)
```

The webcam frame is captured on a background Python thread, processed by `ASCIIEngine` on the GPU, encoded as JPEG, and pushed via a generator to the browser's `<img>` tag as an MJPEG stream. Slider changes travel via WebSocket in the opposite direction.

---

## Layout Structure

```
┌──────────────────────────────────┬──────────────┐
│  HEADER (brand + status)         │              │
├──────────────────────────────────┤   SIDEBAR    │
│                                  │  ──────────  │
│         VIDEO VIEWPORT           │  Camera      │
│   (MJPEG stream, 16:9 aspect)    │  Edge Detect │
│                                  │  Luminance   │
│   [Fullscreen] [Snapshot] overlay│  Toggles     │
├──────────────────────────────────┤  View Mode   │
│  THEME BAR (5 presets)           │              │
└──────────────────────────────────┴──────────────┘
```

---

## Slider Parameter Contracts

All values are transmitted as their true mathematical floats (no integer scaling). The browser maps HTML integer ticks back to real floats before sending.

| Parameter | JS Key | Min | Max | Step | Default | Backend Type |
|:---|:---|:---|:---|:---|:---|:---|
| Zoom | `zoom` | 1.0 | 5.0 | 0.1 | 1.0 | float |
| Offset X | `offset_x` | -1.0 | 1.0 | 0.05 | 0.0 | float |
| Offset Y | `offset_y` | -1.0 | 1.0 | 0.05 | 0.0 | float |
| Kernel Size | `kernel_size` | 1 | 10 | 1 | 2 | int |
| Sigma | `sigma` | 0.1 | 5.0 | 0.1 | 2.0 | float |
| Sigma Scale | `sigma_scale` | 0.1 | 5.0 | 0.1 | 1.6 | float |
| Tau | `tau` | 0.0 | 2.0 | 0.05 | 1.0 | float |
| Threshold | `threshold` | 0.001 | 0.1 | 0.001 | 0.005 | float |
| Edge Votes | `edge_threshold` | 0 | 64 | 1 | 8 | int |
| Exposure | `exposure` | 0.1 | 5.0 | 0.1 | 1.0 | float |
| Attenuation | `attenuation` | 0.1 | 5.0 | 0.1 | 1.0 | float |
| Color Blend | `blend_with_base` | 0.0 | 1.0 | 0.1 | 0.0 | float |
| Intensity | `intensity` | 0.5 | 5.0 | 0.1 | 2.0 | float |

> **Note on Zoom:** Minimum is clamped to `1.0` to prevent image shrinking and border replication artifacts. Going below 1× would scale the image smaller than the viewport, causing OpenCV's `BORDER_REPLICATE` mode to fill edges with repeated pixel rows.

---

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws`

**Message format:** JSON object with any subset of parameter keys:

```json
{ "zoom": 1.5, "sigma": 2.3, "draw_edges": true }
```

The server updates only the keys present in the message, leaving all others at their current values. No acknowledgement message is sent back — this is fire-and-forget for minimum latency.

---

## Fullscreen Mode

The HTML5 Fullscreen API is used. Triggered by:
- Double-clicking the video viewport
- Clicking the fullscreen icon button overlay
- Pressing `F` on the keyboard

In fullscreen mode, the sidebar, header, and theme bar are hidden via the CSS `:fullscreen` selector. The video fills the monitor completely.

---

## Design System

| Token | Value |
|:---|:---|
| Background | `#0a0a0f` |
| Surface (glass) | `rgba(255,255,255,0.04)` |
| Border | `rgba(255,255,255,0.08)` |
| Accent | `#4fc3f7` (cyan blue) |
| Success/On | `#69f0ae` (mint green) |
| Typography | Outfit (UI), Inter (mono values) |
| Border radius | 8px / 14px / 20px |
| Backdrop blur | 8–16px (glassmorphism) |
