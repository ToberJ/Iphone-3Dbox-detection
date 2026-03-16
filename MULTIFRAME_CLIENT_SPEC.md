# Multi-Frame 3D Detection API - Client Specification

## Endpoint

```
POST /detect3d_json
Content-Type: application/json
```

Same endpoint for single-frame and multi-frame. Multi-frame is triggered by the presence of the `frames` key.

---

## Request Format

```json
{
  "frames": [
    {
      "image_base64": "<base64 PNG, RGB>",
      "depth_base64": "<base64 16-bit PNG, millimeters>",
      "intrinsic": {
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      },
      "camera_to_world": {
        "matrix_4x4": [
          [R00, R01, R02, tx],
          [R10, R11, R12, ty],
          [R20, R21, R22, tz],
          [0,   0,   0,   1 ]
        ]
      }
    },
    ...
  ],
  "text_prompt": "monitor.keyboard.table.chair",
  "score_threshold": 0.3,
  "depth_align": true,
  "source": "iphone"
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frames` | array | Yes | Array of frame objects (5-15 frames recommended) |
| `frames[i].image_base64` | string | Yes | Base64-encoded RGB PNG |
| `frames[i].depth_base64` | string | No | Base64-encoded 16-bit grayscale PNG, values in **millimeters** |
| `frames[i].intrinsic.K` | float[3][3] | Yes | Camera intrinsic matrix for this frame |
| `frames[i].camera_to_world.matrix_4x4` | float[4][4] | Yes | 4x4 camera-to-world transform for this frame |
| `text_prompt` | string | Yes | Dot-separated object categories |
| `score_threshold` | float | No | Confidence threshold 0-1 (default: 0.3) |
| `depth_align` | bool | No | Enable depth-based alignment (default: false) |
| `source` | string | No | `"iphone"` or `"quest"` (default: `"quest"`) |

### Naming Convention

Use exactly these nested key names:
- `intrinsic.K` (NOT `intrinsic_K`)
- `camera_to_world.matrix_4x4` (NOT `camera_to_world` as a flat array)

---

## Image Requirements

| Property | Value |
|----------|-------|
| Format | PNG, base64 encoded |
| Color | RGB, 8-bit per channel |
| Resolution | Any (internally resized to 1008x1008) |
| Recommended | Up to 1920px long edge |

---

## Depth Requirements

| Property | Value |
|----------|-------|
| Format | PNG, base64 encoded |
| Bit depth | **16-bit grayscale** (`uint16`) |
| Unit | **Millimeters** (value 1500 = 1.5 meters) |
| Invalid pixels | Value = 0 |
| Resolution | Any (resized to match RGB internally) |
| Alignment | Must be aligned to RGB camera intrinsics |

If `depth_base64` is omitted for a frame, the API uses monocular depth estimation for that frame.

---

## Intrinsic Matrix K

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

**cy convention:**
- iPhone (`source: "iphone"`): send cy in **bottom-left** convention (convert from ARKit top-left before sending: `cy_send = image_height - cy_arkit`)
- Quest (`source: "quest"`): send cy in **bottom-left** convention

The API internally converts to OpenCV top-left convention.

---

## Camera-to-World Matrix

4x4 row-major homogeneous transform. Must be wrapped in `{"matrix_4x4": [...]}`:

```json
"camera_to_world": {
  "matrix_4x4": [
    [0.882, 0.135, 0.451, -0.020],
    [-0.029, 0.972, -0.234, 0.771],
    [-0.470, 0.193, 0.861, -0.230],
    [0, 0, 0, 1]
  ]
}
```

All frames must be in the same world coordinate system (e.g., ARKit session origin or Quest guardian origin).

---

## Capture Recommendations

| Parameter | Recommended |
|-----------|-------------|
| Duration | 1-3 seconds |
| Frame rate | 5 FPS |
| Number of frames | 5-15 |
| Camera motion | Slow pan/orbit around the scene |
| Scene | Static (no moving objects) |

More frames = more robust, but slower inference. 5 frames is a good trade-off for speed. 15 frames gives best accuracy.

---

## Response Format

```json
{
  "boxes": [
    {
      "label": "monitor",
      "center": [3.75, 0.20, 2.91],
      "size": [0.06, 0.37, 0.58],
      "rotation": [0.01, 0.71, -0.01, 0.71],
      "color": [0.0, 1.0, 0.0],
      "score": 0.74,
      "n_frames": 15
    }
  ],
  "mode": "multiview+rgbd+align"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `center` | float[3] | `[x, y, z]` world coordinates (meters) |
| `size` | float[3] | `[width, height, depth]` (meters) |
| `rotation` | float[4] | Quaternion `[qx, qy, qz, qw]` |
| `label` | string | Object category |
| `score` | float | Average confidence across frames (0-1) |
| `color` | float[3] | Display color `[r, g, b]` (0-1) |
| `n_frames` | int | Number of frames this object was detected in |
| `mode` | string | Processing mode used |

### Mode Values

| Mode | Description |
|------|-------------|
| `mono` | Single frame, monocular depth |
| `rgbd` | Single frame, LiDAR depth |
| `rgbd+align` | Single frame, LiDAR + alignment |
| `multiview` | Multi-frame, monocular |
| `multiview+rgbd` | Multi-frame, LiDAR depth |
| `multiview+rgbd+align` | Multi-frame, LiDAR + alignment |

### Filtering by `n_frames`

`n_frames` indicates detection consistency. Higher = more reliable:
- `n_frames >= 10` (out of 15): very reliable
- `n_frames >= 5`: reliable
- `n_frames = 2-3`: borderline, may be false positive

The client can optionally filter by `n_frames` for additional robustness.

---

## Single-Frame (Backward Compatible)

Omit the `frames` key and use `image_base64` directly:

```json
{
  "image_base64": "<base64 PNG>",
  "depth_base64": "<base64 16-bit PNG>",
  "intrinsic": {"K": [...]},
  "camera_to_world": {"matrix_4x4": [...]},
  "text_prompt": "monitor.table",
  "score_threshold": 0.3,
  "source": "iphone"
}
```

---

## Example: Building the Request (Python)

```python
import base64, json, requests

frames = []
for i in range(15):
    with open(f"frame_{i:02d}_rgb.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    with open(f"frame_{i:02d}_depth.png", "rb") as f:
        depth_b64 = base64.b64encode(f.read()).decode()

    frames.append({
        "image_base64": img_b64,
        "depth_base64": depth_b64,
        "intrinsic": {"K": per_frame_K[i]},
        "camera_to_world": {"matrix_4x4": per_frame_c2w[i]},
    })

payload = {
    "frames": frames,
    "text_prompt": "monitor.keyboard.table.chair",
    "score_threshold": 0.3,
    "depth_align": True,
    "source": "iphone",
}

url = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"
headers = {"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"}
resp = requests.post(url, headers=headers, json=payload, timeout=120)
print(resp.json())
```

## Example: Building the Request (Swift)

```swift
var framesArray: [[String: Any]] = []

for i in 0..<capturedFrames.count {
    let frame = capturedFrames[i]

    let imgData = frame.rgbImage.pngData()!
    let imgB64 = imgData.base64EncodedString()

    var frameDict: [String: Any] = [
        "image_base64": imgB64,
        "intrinsic": ["K": frame.intrinsicK],
        "camera_to_world": ["matrix_4x4": frame.cameraToWorld4x4],
    ]

    if let depthData = frame.depthPNG {
        frameDict["depth_base64"] = depthData.base64EncodedString()
    }

    framesArray.append(frameDict)
}

let payload: [String: Any] = [
    "frames": framesArray,
    "text_prompt": "monitor.keyboard.table.chair",
    "score_threshold": 0.3,
    "depth_align": true,
    "source": "iphone",
]
```

---

## Performance

| Frames | Approx. inference time | Accuracy |
|--------|----------------------|----------|
| 1 (single) | ~3s per category | Baseline |
| 5 | ~5s total | Good |
| 15 | ~6s total | Best |

Multi-frame is fast because the backbone features are cached. Most time is spent on the first frame.

---

## Timeout

Set client timeout to **120 seconds** for multi-frame requests. The payload can be 30-50 MB for 15 frames with depth.
