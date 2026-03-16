# Multi-Frame Video Capture API Specification

The API supports an optional multi-frame input mode for improved 3D detection accuracy. Instead of a single image, the client sends multiple frames captured over a short time window (e.g., 1 second at 5 FPS), each with its own camera pose. The API can use multi-view geometry to produce better 3D bounding boxes.

---

## Endpoint

```
POST /detect3d_json
```

Same endpoint as single-frame. The API distinguishes single vs. multi-frame by the presence of the `frames` key.

---

## Request Format (Multi-Frame)

```json
{
  "frames": [
    {
      "image_base64": "<base64 PNG, RGB>",
      "depth_base64": "<base64 16-bit PNG, millimeters, optional>",
      "intrinsic": {
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      },
      "camera_to_world": {
        "matrix_4x4": [[...], [...], [...], [...]]
      }
    }
  ],
  "text_prompt": "monitor.table.chair",
  "score_threshold": 0.3,
  "depth_align": true,
  "source": "iphone"
}
```

### Frame Array

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frames` | array | Yes (for multi-frame) | Array of frame objects, ordered chronologically |
| `frames[i].image_base64` | string | Yes | Base64-encoded PNG (RGB) |
| `frames[i].depth_base64` | string | No | Base64-encoded 16-bit PNG (millimeters). If absent, monocular fallback per frame. |
| `frames[i].intrinsic.K` | 3x3 float array | Yes | Camera intrinsic matrix for this frame |
| `frames[i].camera_to_world.matrix_4x4` | 4x4 float array | Yes | Camera-to-world transform for this frame |

### Shared Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text_prompt` | string | Required | Dot-separated class names |
| `score_threshold` | float | 0.3 | Minimum detection confidence |
| `depth_align` | bool | false | Enable front-surface depth alignment |
| `source` | string | "quest" | `"iphone"` or `"quest"` (controls cy convention) |

### Typical Usage

- **5 frames at 5 FPS** (1 second) or **15 frames at 5 FPS** (3 seconds)
- Each frame is ~1920x1440 resized to keep payload manageable
- Total payload: ~5 MB per frame (image) + ~40 KB per frame (depth) + metadata
- Intrinsics may vary slightly between frames (ARKit auto-focus)
- Camera poses change between frames as the user moves
- More frames + wider baseline = better multi-view triangulation

---

## Single-Frame (Backward Compatible)

Existing single-frame requests remain unchanged:

```json
{
  "image_base64": "...",
  "depth_base64": "...",
  "intrinsic": { "K": [...] },
  "camera_to_world": { "matrix_4x4": [...] },
  "text_prompt": "...",
  "source": "iphone"
}
```

The API detects single-frame mode when `frames` key is absent and `image_base64` is present.

---

## Response Format

Same for both single and multi-frame:

```json
{
  "boxes": [
    {
      "center": [x, y, z],
      "size": [sx, sy, sz],
      "rotation": [qx, qy, qz, qw],
      "label": "monitor",
      "score": 0.89,
      "color": [0.0, 1.0, 0.0]
    }
  ],
  "mode": "multiview+rgbd+align"
}
```

### Mode Field Values

| mode | Description |
|------|-------------|
| `mono` | Single frame, monocular depth |
| `rgbd` | Single frame, LiDAR depth |
| `RGB-D+align` | Single frame, LiDAR + alignment |
| `multiview` | Multi-frame, monocular depth |
| `multiview+rgbd` | Multi-frame, LiDAR depth |
| `multiview+rgbd+align` | Multi-frame, LiDAR + alignment |

---

## Camera Pose Convention

All `camera_to_world` matrices follow the same convention as single-frame:

- **Left-handed** (Unity convention) — the iPhone client converts from ARKit right-handed before sending
- **Y-up**, camera looks along **+Z**
- 4x4 matrix: upper-left 3x3 is rotation, right column is translation, bottom row is `[0,0,0,1]`

The poses are in a consistent world coordinate system (ARKit world origin). Frame 0 and frame 4 will have different poses if the user moved the phone during the 1-second capture.

---

## iPhone Client Capture Details

| Property | Value |
|----------|-------|
| Capture duration | 1-3 seconds (configurable) |
| Frame rate | 5 FPS (200ms interval) |
| Number of frames | 5-15 (depends on duration) |
| Image resolution | Up to 1920px (long edge) |
| Depth resolution | 256x192 (native LiDAR) |
| Depth format | 16-bit PNG, millimeters |
| Intrinsics | Per-frame (may vary with auto-focus) |
| Camera pose | Per-frame, ARKit world coordinates converted to LH |

---

## Example: 3-Second Capture with Trajectory (15 frames at 5 FPS)

This example shows a user slowly panning the iPhone to the right over 3 seconds while looking at a desk with monitors. Image and depth base64 strings are abbreviated; in practice they are full base64-encoded PNGs.

```json
{
  "frames": [
    {
      "image_base64": "<frame_00.png base64, ~4MB>",
      "depth_base64": "<depth_00.png base64, ~40KB>",
      "intrinsic": { "K": [[1354.16, 0, 957.42], [0, 1354.16, 716.07], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9987,  0.0069,  0.0508, -0.020],
        [-0.0031,  0.9972, -0.0748,  0.091],
        [-0.0511,  0.0745,  0.9959, -0.849],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_01.png base64>",
      "depth_base64": "<depth_01.png base64>",
      "intrinsic": { "K": [[1354.20, 0, 957.45], [0, 1354.20, 716.10], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9983,  0.0072,  0.0573, -0.015],
        [-0.0035,  0.9971, -0.0752,  0.092],
        [-0.0576,  0.0748,  0.9955, -0.847],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_02.png base64>",
      "depth_base64": "<depth_02.png base64>",
      "intrinsic": { "K": [[1354.23, 0, 957.48], [0, 1354.23, 716.08], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9976,  0.0075,  0.0689, -0.008],
        [-0.0040,  0.9970, -0.0760,  0.093],
        [-0.0693,  0.0755,  0.9947, -0.844],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_03.png base64>",
      "depth_base64": "<depth_03.png base64>",
      "intrinsic": { "K": [[1354.18, 0, 957.50], [0, 1354.18, 716.12], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9965,  0.0078,  0.0832,  0.001],
        [-0.0045,  0.9969, -0.0771,  0.094],
        [-0.0836,  0.0764,  0.9936, -0.840],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_04.png base64>",
      "depth_base64": "<depth_04.png base64>",
      "intrinsic": { "K": [[1354.22, 0, 957.44], [0, 1354.22, 716.05], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9951,  0.0082,  0.0985,  0.013],
        [-0.0050,  0.9967, -0.0783,  0.095],
        [-0.0989,  0.0775,  0.9921, -0.835],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_05.png base64>",
      "depth_base64": "<depth_05.png base64>",
      "intrinsic": { "K": [[1354.25, 0, 957.41], [0, 1354.25, 716.09], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9932,  0.0085,  0.1161,  0.028],
        [-0.0056,  0.9965, -0.0796,  0.096],
        [-0.1165,  0.0784,  0.9901, -0.829],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_06.png base64>",
      "depth_base64": "<depth_06.png base64>",
      "intrinsic": { "K": [[1354.19, 0, 957.47], [0, 1354.19, 716.11], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9910,  0.0088,  0.1335,  0.045],
        [-0.0062,  0.9963, -0.0809,  0.097],
        [-0.1339,  0.0794,  0.9878, -0.822],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_07.png base64>",
      "depth_base64": "<depth_07.png base64>",
      "intrinsic": { "K": [[1354.21, 0, 957.43], [0, 1354.21, 716.07], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9884,  0.0091,  0.1513,  0.064],
        [-0.0068,  0.9960, -0.0822,  0.098],
        [-0.1517,  0.0803,  0.9850, -0.814],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_08.png base64>",
      "depth_base64": "<depth_08.png base64>",
      "intrinsic": { "K": [[1354.24, 0, 957.46], [0, 1354.24, 716.10], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9854,  0.0094,  0.1695,  0.085],
        [-0.0075,  0.9957, -0.0836,  0.099],
        [-0.1699,  0.0811,  0.9819, -0.805],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_09.png base64>",
      "depth_base64": "<depth_09.png base64>",
      "intrinsic": { "K": [[1354.17, 0, 957.49], [0, 1354.17, 716.06], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9820,  0.0097,  0.1880,  0.108],
        [-0.0081,  0.9954, -0.0850,  0.100],
        [-0.1884,  0.0820,  0.9784, -0.795],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_10.png base64>",
      "depth_base64": "<depth_10.png base64>",
      "intrinsic": { "K": [[1354.20, 0, 957.42], [0, 1354.20, 716.08], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9781,  0.0100,  0.2067,  0.133],
        [-0.0088,  0.9950, -0.0864,  0.101],
        [-0.2071,  0.0827,  0.9748, -0.784],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_11.png base64>",
      "depth_base64": "<depth_11.png base64>",
      "intrinsic": { "K": [[1354.22, 0, 957.45], [0, 1354.22, 716.11], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9739,  0.0103,  0.2264,  0.160],
        [-0.0094,  0.9946, -0.0879,  0.102],
        [-0.2268,  0.0835,  0.9704, -0.772],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_12.png base64>",
      "depth_base64": "<depth_12.png base64>",
      "intrinsic": { "K": [[1354.18, 0, 957.48], [0, 1354.18, 716.09], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9692,  0.0106,  0.2460,  0.189],
        [-0.0101,  0.9942, -0.0894,  0.103],
        [-0.2463,  0.0842,  0.9656, -0.759],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_13.png base64>",
      "depth_base64": "<depth_13.png base64>",
      "intrinsic": { "K": [[1354.21, 0, 957.44], [0, 1354.21, 716.07], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9641,  0.0109,  0.2652,  0.220],
        [-0.0108,  0.9937, -0.0909,  0.104],
        [-0.2656,  0.0848,  0.9604, -0.745],
        [ 0,       0,       0,       1    ]
      ]}
    },
    {
      "image_base64": "<frame_14.png base64>",
      "depth_base64": "<depth_14.png base64>",
      "intrinsic": { "K": [[1354.23, 0, 957.41], [0, 1354.23, 716.10], [0, 0, 1]] },
      "camera_to_world": { "matrix_4x4": [
        [ 0.9586,  0.0112,  0.2847,  0.253],
        [-0.0115,  0.9932, -0.0924,  0.105],
        [-0.2850,  0.0853,  0.9548, -0.730],
        [ 0,       0,       0,       1    ]
      ]}
    }
  ],
  "text_prompt": "monitor.keyboard.table.chair.computer",
  "score_threshold": 0.3,
  "depth_align": true,
  "source": "iphone"
}
```

### Trajectory Visualization

The camera starts facing roughly forward and pans ~16 degrees to the right over 3 seconds:

```
Frame  Time(s)  Yaw(deg)  tx(m)    tz(m)     Description
-----  -------  --------  ------   -------   -----------
  00     0.0      2.9     -0.020   -0.849    Start position
  01     0.2      3.3     -0.015   -0.847    Slight pan right
  02     0.4      3.9     -0.008   -0.844
  03     0.6      4.8      0.001   -0.840
  04     0.8      5.6      0.013   -0.835
  05     1.0      6.7      0.028   -0.829    1 second mark
  06     1.2      7.7      0.045   -0.822
  07     1.4      8.7      0.064   -0.814
  08     1.6      9.8      0.085   -0.805
  09     1.8     10.8      0.108   -0.795
  10     2.0     11.9      0.133   -0.784    2 second mark
  11     2.2     13.1      0.160   -0.772
  12     2.4     14.3      0.189   -0.759
  13     2.6     15.4      0.220   -0.745
  14     2.8     16.5      0.253   -0.730    End position
```

Key observations:
- **Rotation**: Upper-left 3x3 of each matrix shows gradual yaw rotation (~1 deg per frame)
- **Translation**: tx increases (moving right), tz decreases (moving slightly forward)
- **ty**: Nearly constant (~0.09-0.10m) — user holding phone at same height
- **Intrinsics**: Slight per-frame variation due to ARKit auto-focus
- **All poses share the same world origin** (ARKit session origin)

---

## Implementation Notes

1. **Timeout**: Multi-frame requests are larger and may take longer. Client uses 120s timeout.
2. **Payload size**: ~25-30 MB for 5 frames. Server should accept large request bodies.
3. **Frame ordering**: Frames are in chronological order (frame 0 = earliest).
4. **Degenerate case**: If `frames` has only 1 element, treat as single-frame.
