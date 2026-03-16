# Depth Input Specification for SAM3_3D API

The API now supports optional LiDAR/depth input alongside RGB for improved 3D detection accuracy. When depth is provided, the model uses real depth measurements instead of its internal monocular depth estimation.

---

## Endpoint

```
POST /detect3d_json
```

---

## Request Format

```json
{
  "image_base64": "<base64 PNG, RGB>",
  "depth_base64": "<base64 16-bit PNG, millimeters>",
  "intrinsic": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  "camera_to_world": {
    "matrix_4x4": [[...], [...], [...], [...]]
  },
  "text_prompt": "monitor.table.chair",
  "score_threshold": 0.3,
  "source": "iphone"
}
```

New fields:
- **`depth_base64`** (optional) - If absent, falls back to monocular depth estimation
- **`source`** (optional) - `"iphone"` or `"quest"` (default: `"quest"`)

---

## Depth Map Requirements

| Property | Requirement |
|----------|-------------|
| **Format** | PNG, base64 encoded |
| **Bit depth** | **16-bit grayscale** (single channel, `uint16`) |
| **Unit** | **Millimeters** (1 pixel value = 1 mm) |
| **Invalid pixels** | Value = **0** (no depth data) |
| **Valid range** | Typically 200-5000 (0.2m - 5.0m) |
| **Resolution** | Any resolution (API will resize to match RGB) |
| **Alignment** | **Must be aligned to RGB camera intrinsics** |
| **Origin** | Top-left (standard PNG convention) |

### Important Notes

1. **Must be 16-bit**: 8-bit depth will be rejected. The API validates `dtype == uint16`.
2. **Millimeters, not meters**: A pixel value of `1500` means 1.5 meters.
3. **0 = invalid**: Pixels with value 0 are treated as missing depth (the model will estimate depth for those regions).
4. **Aligned to RGB intrinsics**: The depth map must correspond to the same camera view as the RGB image. The same intrinsic matrix K applies to both. If resolutions differ, the API handles the scaling automatically.

---

## How to Prepare Depth from iPhone LiDAR (ARKit)

### Step 1: Get depth from ARFrame

```swift
guard let sceneDepth = frame.sceneDepth else { return }
let depthMap = sceneDepth.depthMap  // CVPixelBuffer, Float32, meters
// Native resolution: 256 x 192
```

### Step 2: Convert to 16-bit PNG (millimeters)

```swift
func depthToUInt16PNG(depthMap: CVPixelBuffer) -> Data? {
    CVPixelBufferLockBaseAddress(depthMap, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

    let width = CVPixelBufferGetWidth(depthMap)    // 256
    let height = CVPixelBufferGetHeight(depthMap)   // 192
    let baseAddress = CVPixelBufferGetBaseAddress(depthMap)!
    let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)

    // Convert Float32 meters -> UInt16 millimeters
    var uint16Data = [UInt16](repeating: 0, count: width * height)
    for i in 0 ..< width * height {
        let meters = floatBuffer[i]
        if meters > 0 && meters < 65.535 {  // Valid range check
            uint16Data[i] = UInt16(meters * 1000.0)  // meters -> mm
        }
        // else: stays 0 (invalid)
    }

    // Create 16-bit grayscale CGImage
    let bitsPerComponent = 16
    let bytesPerRow = width * 2  // 2 bytes per uint16
    let colorSpace = CGColorSpaceCreateDeviceGray()

    let data = Data(bytes: uint16Data, count: uint16Data.count * 2)
    guard let provider = CGDataProvider(data: data as CFData),
          let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
          ) else { return nil }

    // Encode to PNG
    let uiImage = UIImage(cgImage: cgImage)
    return uiImage.pngData()
}
```

### Step 3: Base64 encode and send

```swift
if let depthPNG = depthToUInt16PNG(depthMap: sceneDepth.depthMap) {
    let depthBase64 = depthPNG.base64EncodedString()

    // Add to request payload
    payload["depth_base64"] = depthBase64
}
```

### Alternative: Using vImage / Accelerate (simpler)

```swift
import Accelerate

func depthToUInt16Data(depthMap: CVPixelBuffer) -> [UInt16] {
    CVPixelBufferLockBaseAddress(depthMap, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

    let w = CVPixelBufferGetWidth(depthMap)
    let h = CVPixelBufferGetHeight(depthMap)
    let ptr = CVPixelBufferGetBaseAddress(depthMap)!
        .assumingMemoryBound(to: Float32.self)

    // Float32 meters -> multiply by 1000 -> clamp -> UInt16
    var floats = Array(UnsafeBufferPointer(start: ptr, count: w * h))
    var scale: Float = 1000.0
    vDSP_vsmul(&floats, 1, &scale, &floats, 1, vDSP_Length(w * h))

    var result = [UInt16](repeating: 0, count: w * h)
    vDSP_vfixu16(floats, 1, &result, 1, vDSP_Length(w * h))
    return result
}
```

---

## `source` Field

The `source` field tells the API which cy convention to use:

| Source | cy convention received | API action |
|--------|----------------------|------------|
| `"iphone"` | Bottom-left (client already converted from ARKit top-left) | `cy_opencv = H - cy` |
| `"quest"` | Bottom-left (client already converted from Unity) | `cy_opencv = H - cy` |
| `"opencv"` | Already OpenCV top-left | No conversion |

**For iPhone**: set `"source": "iphone"` and convert cy before sending:
```swift
let cy_send = Float(imageHeight) - cy_arkit  // top-left -> bottom-left
```

---

## Full iPhone Example (Python equivalent)

```python
import requests, json, base64, cv2
import numpy as np

url = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"
headers = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json"
}

# Load RGB
with open("capture.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Load depth (16-bit PNG, millimeters)
with open("depth.png", "rb") as f:
    depth_b64 = base64.b64encode(f.read()).decode()

# Verify depth format
depth_check = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
assert depth_check.dtype == np.uint16, f"Must be 16-bit, got {depth_check.dtype}"
print(f"Depth: {depth_check.shape}, range: {depth_check.min()}-{depth_check.max()} mm")

payload = {
    "image_base64": img_b64,
    "depth_base64": depth_b64,
    "intrinsic": {
        "K": [[1344.43, 0, 957.86],
              [0, 1344.43, 724.37],  # cy: bottom-left convention
              [0, 0, 1]]
    },
    "camera_to_world": {
        "matrix_4x4": [
            [1, 0, 0, 0],
            [0, 1, 0, 1.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "text_prompt": "monitor.table.chair.laptop",
    "score_threshold": 0.3,
    "source": "iphone"
}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
boxes = resp.json()["boxes"]
for b in boxes:
    print(f"{b['label']}: score={b['score']:.2f}, center={b['center']}")
```

---

## Validation

The API validates the depth input:
1. **Decode check**: Must be a valid PNG image
2. **Data type check**: Must be `uint16` (16-bit). 8-bit PNG will be rejected.
3. **Auto-resize**: Any resolution is accepted. API resizes to match RGB using nearest-neighbor interpolation.
4. **Logging**: Server prints depth stats (shape, range, valid pixel %) for debugging.

---

## Without Depth (Monocular Fallback)

Simply omit the `depth_base64` field. The API will use its built-in monocular depth estimation (LingBot-Depth). Everything else stays the same.

```json
{
  "image_base64": "<base64 PNG>",
  "intrinsic": {"K": [...]},
  "camera_to_world": {"matrix_4x4": [...]},
  "text_prompt": "chair",
  "score_threshold": 0.3,
  "source": "iphone"
}
```
