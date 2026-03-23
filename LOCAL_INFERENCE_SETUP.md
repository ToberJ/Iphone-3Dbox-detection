# Local ONNX Runtime Inference Setup Guide

Run SAM3_3D 3D object detection locally on iPhone — no server needed.

## Prerequisites

- Mac with Xcode 15+
- iPhone 15 Pro (recommended, has LiDAR + Neural Engine)
- CocoaPods installed (`sudo gem install cocoapods`)

## Step 1: Install ONNX Runtime via CocoaPods

```bash
cd iphone_demo
pod init
```

Edit the generated `Podfile`:

```ruby
platform :ios, '16.0'

target 'iPhone3DDetect' do
  use_frameworks!
  pod 'onnxruntime-objc'
end
```

Install:

```bash
pod install
```

**From now on, always open `iPhone3DDetect.xcworkspace` (NOT `.xcodeproj`):**

```bash
open iPhone3DDetect.xcworkspace
```

## Step 2: Add Import to Swift Files

In `LocalInferenceService.swift`, the import is already there:

```swift
import onnxruntime_objc
```

If Xcode shows "No such module", make sure you opened the `.xcworkspace` file.

## Step 3: Build & Run

1. Open `iPhone3DDetect.xcworkspace` in Xcode
2. Select your iPhone as the target device
3. Product → Build (Cmd+B)
4. Product → Run (Cmd+R)

## Step 4: Switch to Local Mode

1. In the app, tap the **gear icon** (Settings)
2. Tap **"Switch to Local (ONNX)"**
3. First time: the app will download the model from HuggingFace (~6GB)
   - `model.onnx` (15MB) — graph structure
   - `weights.bin` (5.9GB) — model weights
   - Stored in app's Documents directory, only downloaded once
4. After download, tap **Detect** — inference runs locally on device

## How It Works

```
Same as before:                    New local mode:
iPhone → HTTP → Server → Response  iPhone → ONNX Runtime → Response
         ↓                                   ↓
   DetectionService.swift           LocalInferenceService.swift
```

Both produce the same `DetectionResponse`. The app doesn't need to know which backend is being used.

### Pipeline

```
1. ARKit capture (RGB + LiDAR depth + camera intrinsics + pose)
2. Preprocessing (Swift):
   - fix_cy: convert ARKit cy to OpenCV convention
   - Resize image to fit 1008x1008
   - Scale intrinsics by resize ratio
   - ImageNet normalize
   - Center pad to 1008x1008
   - Adjust intrinsics for padding
   - Resize + pad depth map (nearest interpolation)
3. ONNX Runtime inference (Neural Engine when available):
   - Input: image (1,3,1008,1008) + depth (1,1,1008,1008) + intrinsics (1,3,3)
   - Output: raw logits + boxes + 3D params + presence + depth_map + K_pred
4. Postprocessing (Swift):
   - Sigmoid scores × presence score
   - Per-class NMS
   - 3D box decode (delta center, depth, dims, 6D rotation → quaternion)
   - Canonical rotation normalization
   - Camera-to-world transform with Y-flip
   - Cross-class NMS (0.3m distance)
5. Render 3D boxes in AR
```

## Model Details

- **Checkpoint**: `epoch=2-step=6450.ckpt` (ITW canonical rotation)
- **Parameters**: 1180M (SAM3 ViT-H + LingBot-Depth DINOv2 + 3D Head)
- **Baked text prompts**: monitor, keyboard, table, chair, computer
- **Model URL**: https://huggingface.co/weikaih/iphone_test

## Performance

- **Neural Engine (ANE)**: ONNX Runtime automatically offloads compatible ops to ANE via CoreML EP
- **Expected inference**: ~1-3 seconds per frame on iPhone 15 Pro
- **Memory**: ~3-4GB runtime (fits in iPhone 15 Pro's 8GB)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No such module 'onnxruntime_objc'" | Open `.xcworkspace` not `.xcodeproj` |
| Build fails with signing error | Xcode → Signing & Capabilities → select your Team |
| Model download stalls | Check WiFi. Model is 6GB. Can take several minutes. |
| App crashes on launch with OOM | Device needs 8GB RAM (iPhone 15 Pro+) |
| CoreML EP warning in console | Normal — some ops fall back to CPU, performance still fine |
| Detection results differ from server | Expected — slight preprocessing differences (resize interpolation) |

## Switching Back to Server

Settings → **"Switch to Server"** — goes back to HTTP API mode instantly.

## Files

| File | Description |
|------|-------------|
| `LocalInferenceService.swift` | ONNX Runtime inference + pre/post processing |
| `ViewController.swift` | UI toggle for local/server mode |
| `DetectionService.swift` | HTTP API client (unchanged) |
| `Models.swift` | Shared data structures (BBox3D, DetectionResponse) |
