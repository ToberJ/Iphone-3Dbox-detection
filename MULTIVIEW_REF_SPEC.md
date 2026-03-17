# Multi-View Reference Mode - Client Specification

Use 1 main frame for detection + 1-3 reference frames (depth only or depth+RGB) to improve 3D box depth accuracy via multi-view geometric consistency.

---

## Concept

```
Main Frame (front view):
  - RGB image → run 3D detection → get 3D boxes
  - Depth map → single-view depth constraint

Reference Frame(s) (side/angled views):
  - Depth map + camera pose → additional depth constraints
  - RGB (optional) → SAM2 mask for cleaner depth sampling

Multi-view optimization:
  - Merge point clouds from all views into main camera space
  - Find optimal uniform scale that fits all views' depth
  - 2D projection preserved in main view
```

---

## Endpoint

```
POST /detect3d_json
```

Same endpoint. The API detects multi-view mode when `reference_frames` key is present.

---

## Request Format

```json
{
  "image_base64": "<main frame RGB, base64 PNG>",
  "depth_base64": "<main frame depth, 16-bit PNG mm>",
  "intrinsic": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  "camera_to_world": {
    "matrix_4x4": [[...], [...], [...], [...]]
  },

  "reference_frames": [
    {
      "depth_base64": "<ref depth, 16-bit PNG mm>",
      "image_base64": "<ref RGB, optional>",
      "intrinsic": {
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      },
      "camera_to_world": {
        "matrix_4x4": [[...], [...], [...], [...]]
      }
    }
  ],

  "text_prompt": "monitor.table.chair",
  "score_threshold": 0.5,
  "alignment_post_process": true,
  "source": "iphone"
}
```

### Main Frame Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_base64` | string | Yes | RGB image for detection |
| `depth_base64` | string | No | Depth map (16-bit PNG, mm) |
| `intrinsic.K` | float[3][3] | Yes | Camera intrinsic matrix |
| `camera_to_world.matrix_4x4` | float[4][4] | Yes | Camera-to-world transform |

### Reference Frame Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `depth_base64` | string | Yes | Depth map (16-bit PNG, mm) |
| `image_base64` | string | No | RGB image (enables SAM2 mask in ref view) |
| `intrinsic.K` | float[3][3] | Yes | Camera intrinsic matrix for this view |
| `camera_to_world.matrix_4x4` | float[4][4] | Yes | Camera-to-world transform for this view |

### Shared Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text_prompt` | string | Required | Dot-separated object names |
| `score_threshold` | float | 0.3 | Detection confidence threshold |
| `alignment_post_process` | bool | false | Enable depth refinement |
| `source` | string | "quest" | `"iphone"` or `"quest"` (cy convention) |

---

## How It Works

### Step 1: Detection on Main Frame
- Run SAM3_3D on main frame RGB → 3D boxes in main camera space
- Each box has: center (x,y,z), dimensions (w,l,h), rotation (quaternion)

### Step 2: For Each Box, Extract Point Clouds

**Main frame:**
- SAM2 mask from 2D detection box
- Extract 3D points within mask from main depth map
- Points are in main camera coordinate space

**Each reference frame:**
- Project 3D box corners from main cam → world → ref cam
- Get 2D bounding box of projected corners in ref image
- If ref has RGB: run SAM2 with projected box → mask → extract depth within mask
- If ref has no RGB: extract depth within projected 2D box region
- Transform ref 3D points back to main camera space

### Step 3: Merge + Optimize

```
P_merged = P_main + P_ref1 + P_ref2 + ...

Optimize uniform scale s ∈ [0.8, 1.2]:
  loss(s) = anchor_inclusion_loss + anchor_tightness_loss

  inclusion: penalize points outside the scaled box
  tightness: penalize empty space inside the scaled box

Method: grid search (20 steps) + L-BFGS-B
Result: box[:6] *= optimal_s
```

### Step 4: Quality Checks (Skip Conditions)

A box is NOT refined if any of these are true:
1. **Not thin-facing**: avg(X-extent, Y-extent) < 1.5 * Z-extent in camera space
2. **Overlaps with other box**: 2D IoU > 0.2 with any other detected box
3. **Bad SAM2 mask**: mask_area / box_area < 0.8

Skipped boxes keep their original model prediction.

---

## Coordinate Convention

All `camera_to_world` matrices must be in the **same world coordinate system** across main and reference frames.

| Source | Convention |
|--------|-----------|
| iPhone (ARKit) | Convert to left-handed before sending (client does this) |
| Quest (Unity) | Already left-handed |

The API handles internal coordinate conversions (cy flip, OpenCV ↔ Unity).

---

## Capture Recommendations

### Best Practices

| Parameter | Recommendation |
|-----------|----------------|
| Number of ref frames | 1-3 |
| Angular difference | 30-90 degrees from main view |
| Distance | Similar distance to the scene |
| Overlap | Objects must be visible in both main and ref views |
| Scene | Static (no moving objects between captures) |

### Good View Combinations

```
Main: front view (facing the desk)
Ref 1: side view (45-90 degrees to the right)
Ref 2: opposite side (45-90 degrees to the left)
```

The more different the viewing angle, the stronger the depth constraint. But objects must still be visible in the reference views.

### What NOT to do

- Don't use views where the target object is occluded
- Don't use views from the same angle (no new depth information)
- Don't move objects between captures

---

## Response Format

Same as single-frame:

```json
{
  "boxes": [
    {
      "label": "monitor",
      "center": [x, y, z],
      "size": [w, h, d],
      "rotation": [qx, qy, qz, qw],
      "color": [r, g, b],
      "score": 0.90
    }
  ],
  "mode": "rgbd+sam2_refine"
}
```

---

## Example: iPhone Client (Python)

```python
import requests, json, base64

url = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"
headers = {"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"}

# Main frame
with open("front_rgb.png", "rb") as f:
    main_rgb = base64.b64encode(f.read()).decode()
with open("front_depth.png", "rb") as f:
    main_depth = base64.b64encode(f.read()).decode()

# Reference frame
with open("side_depth.png", "rb") as f:
    ref_depth = base64.b64encode(f.read()).decode()
with open("side_rgb.png", "rb") as f:  # optional
    ref_rgb = base64.b64encode(f.read()).decode()

payload = {
    "image_base64": main_rgb,
    "depth_base64": main_depth,
    "intrinsic": {"K": [[1351.5, 0, 957.1], [0, 1351.5, 715.9], [0, 0, 1]]},
    "camera_to_world": {"matrix_4x4": [
        [0.967, -0.065, -0.247, -0.297],
        [0.003, 0.970, -0.245, -0.016],
        [0.256, 0.236, 0.938, -0.093],
        [0, 0, 0, 1]
    ]},

    "reference_frames": [
        {
            "depth_base64": ref_depth,
            "image_base64": ref_rgb,  # optional, enables SAM2 in ref view
            "intrinsic": {"K": [[1347.2, 0, 957.5], [0, 1347.2, 715.9], [0, 0, 1]]},
            "camera_to_world": {"matrix_4x4": [
                [-0.092, -0.222, -0.971, 1.554],
                [-0.048, 0.975, -0.218, -0.172],
                [0.995, 0.026, -0.101, 1.877],
                [0, 0, 0, 1]
            ]},
        }
    ],

    "text_prompt": "monitor.table.chair",
    "score_threshold": 0.5,
    "alignment_post_process": True,
    "source": "iphone",
}

resp = requests.post(url, headers=headers, json=payload, timeout=120)
for box in resp.json()["boxes"]:
    print(f"{box['label']}: center={[round(x,2) for x in box['center']]}")
```

---

## Example: Swift (Capture Two Views)

```swift
// Capture main frame
let mainCapture = captureFrame()  // RGB + depth + pose + intrinsics

// User moves phone to a different angle
let refCapture = captureFrame()   // RGB + depth + pose + intrinsics

// Build request
let payload: [String: Any] = [
    "image_base64": mainCapture.rgbBase64,
    "depth_base64": mainCapture.depthBase64,
    "intrinsic": ["K": mainCapture.intrinsicK],
    "camera_to_world": ["matrix_4x4": mainCapture.cameraToWorld],

    "reference_frames": [
        [
            "image_base64": refCapture.rgbBase64,
            "depth_base64": refCapture.depthBase64,
            "intrinsic": ["K": refCapture.intrinsicK],
            "camera_to_world": ["matrix_4x4": refCapture.cameraToWorld],
        ]
    ],

    "text_prompt": "monitor.table.chair",
    "score_threshold": 0.5,
    "alignment_post_process": true,
    "source": "iphone",
]
```

---

## Performance

| Configuration | Approx. time |
|--------------|-------------|
| Main only (no ref) | ~3s per category |
| Main + 1 ref (depth only) | ~3s + ~0.1s |
| Main + 1 ref (depth + RGB + SAM2) | ~3s + ~0.5s |
| Main + 3 refs (depth + RGB + SAM2) | ~3s + ~1.5s |

Reference frame processing is fast because no model inference is needed on ref frames — only SAM2 (if RGB provided) + point cloud extraction + coordinate transform.

---

## Timeout

Set client timeout to **120 seconds** for requests with reference frames.
