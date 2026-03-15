# iPhone (ARKit) Client — API Input/Output Specification

This document describes the data and conventions sent by the Swift/ARKit client running on iPhone, compared against the Quest 3 client that the API was originally designed for.

---

## 1. Request Format

Same endpoint and JSON structure as Quest:

```json
{
  "image_base64": "<base64 PNG>",
  "intrinsic": { "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] },
  "camera_to_world": { "matrix_4x4": [[...], [...], [...], [...]] },
  "text_prompt": "monitor.keyboard.table.chair",
  "score_threshold": 0.5
}
```

---

## 2. Image (`image_base64`)

| Property        | iPhone (ARKit)                     | Quest 3 (Unity)                    |
|-----------------|------------------------------------|------------------------------------|
| Source          | ARKit `ARFrame.capturedImage`      | Left passthrough camera            |
| Resolution     | **1920 x 1440** (resized from native) | **1280 x 960**                  |
| Format         | PNG, base64 encoded                | PNG, base64 encoded                |
| Color space    | RGB                                | RGB                                |
| Origin         | **Top-left** (standard PNG)        | **Top-left** (standard PNG)        |
| Orientation    | **Landscape-right** (always, regardless of phone orientation) | Upright |

> ARKit's `capturedImage` is always landscape-right. This matches Quest's landscape capture.

**Status: OK** — No conversion needed.

---

## 3. Intrinsic Matrix K (`intrinsic.K`)

| Parameter | iPhone Value (example) | Source |
|-----------|----------------------|--------|
| `fx`      | 1359.59              | `frame.camera.intrinsics.columns.0.x` |
| `fy`      | 1359.59              | `frame.camera.intrinsics.columns.1.y` |
| `cx`      | 965.86               | `frame.camera.intrinsics.columns.2.x` |
| `cy`      | 717.06               | `frame.camera.intrinsics.columns.2.y` |

### ARKit vs Quest cy convention:

| Convention     | Origin      | Y direction | cx     | cy meaning        |
|----------------|-------------|-------------|--------|-------------------|
| Quest (Unity)  | Bottom-left | Up          | from left | from **bottom** |
| iPhone (ARKit) | **Top-left**| **Down**    | from left | from **top**    |

### BUG #1: cy is NOT converted

The API server was built for Quest and expects cy measured from the **bottom edge** (Unity convention). It internally converts: `cy_internal = H - cy_sent`.

**Current code sends ARKit cy directly (from top edge).** The server double-converts it:
```
Server does: cy_internal = H - cy_sent = 1440 - 717 = 723  (WRONG, should be 717)
```

**Fix needed:**
```swift
let cy_to_send = Float(newH) - cy   // Convert top-edge to bottom-edge
```

This way: server does `H - (H - cy) = cy` → correct.

---

## 4. Camera-to-World Matrix (`camera_to_world.matrix_4x4`)

### Coordinate system comparison:

| Property     | Quest (Unity)         | iPhone (ARKit)       |
|-------------|----------------------|---------------------|
| Handedness  | **LEFT-HANDED**      | **RIGHT-HANDED**    |
| X-axis      | Right (+)            | Right (+)           |
| Y-axis      | Up (+)               | Up (+)              |
| Z-axis      | **Forward (+)**      | **Backward (+)**    |
| Camera forward | **+Z**            | **-Z**              |
| Units       | Meters               | Meters              |

### BUG #2: camera_to_world is not converted from RH to LH

The API expects a **left-handed** camera_to_world matrix. We are sending the raw ARKit **right-handed** matrix.

**Current code (WRONG):**
```swift
let camToWorld: [[Float]] = [
    [t.columns.0.x, t.columns.1.x, t.columns.2.x, t.columns.3.x],  // as-is
    [t.columns.0.y, t.columns.1.y, t.columns.2.y, t.columns.3.y],
    [t.columns.0.z, t.columns.1.z, t.columns.2.z, t.columns.3.z],
    ...
]
```

**Fix needed** (negate Z column + Z row of rotation + tz, preserves det=+1):
```swift
// Row 0: [R00,  R01, -R02,  tx]
// Row 1: [R10,  R11, -R12,  ty]
// Row 2: [-R20, -R21, R22, -tz]
// Row 3: [0,    0,    0,    1 ]
let camToWorld: [[Float]] = [
    [ t.columns.0.x,  t.columns.1.x, -t.columns.2.x,  t.columns.3.x],
    [ t.columns.0.y,  t.columns.1.y, -t.columns.2.y,  t.columns.3.y],
    [-t.columns.0.z, -t.columns.1.z,  t.columns.2.z, -t.columns.3.z],
    [0, 0, 0, 1],
]
```

This is the standard RH↔LH conversion: `M_lh = S * M_rh * S` where `S = diag(1,1,-1)`.
Determinant of rotation part stays +1 (no 500 error).

---

## 5. Response Handling (Rendering)

The API returns boxes in **left-handed** (Unity) world coordinates:
- `center`: [x, y, z] in left-handed, meters
- `rotation`: [qx, qy, qz, qw] quaternion, left-handed, scalar-last

To render in ARKit (right-handed), convert back:

**Position:** negate Z
```swift
simd_float3(center.x, center.y, -center.z)
```

**Quaternion:** negate ix, iy (Z-mirror transform)
```swift
simd_quatf(ix: -q.imag.x, iy: -q.imag.y, iz: q.imag.z, r: q.real)
```

**Status: Current code is correct** for rendering conversion.

---

## 6. Summary of Bugs Found

| Issue | Current State | Fix |
|-------|-------------|-----|
| **cy convention** | Sending ARKit top-edge cy directly | Convert: `cy_to_send = H - cy` |
| **camera_to_world handedness** | Sending raw RH matrix | Convert RH→LH: negate Z col + Z row + tz |
| **Rendering position** | `(x, y, -z)` | Correct ✓ |
| **Rendering rotation** | `(-qx, -qy, qz, qw)` | Correct ✓ |
