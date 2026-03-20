import Foundation
import CoreML
import Vision
import Accelerate
import simd

/// On-device SAM3_3D inference using CoreML models.
/// Provides the same interface as DetectionService but runs locally.
///
/// Models required in app bundle:
///   - sam3_3d_unified_int8.mlpackage  (backbone + encoder + decoder, ~486MB)
///   - head3d_int8.mlpackage           (3D regression head, ~3MB)
///   - text_embeddings.bin             (pre-computed text features, ~40KB)
class LocalInferenceService {
    static let shared = LocalInferenceService()

    private var unifiedModel: MLModel?
    private var head3dModel: MLModel?
    private var textFeatures: MLMultiArray?
    private var textMask: MLMultiArray?
    private(set) var isModelLoaded = false
    private var isLoaded = false

    // Same text prompts as server (dot-separated)
    var textPrompt: String = "monitor.keyboard.table.chair.computer"
    var scoreThreshold: Float = 0.5

    // SAM3 input size
    private let inputSize: Int = 1008

    // ImageNet normalization
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]

    private init() {}

    // MARK: - Model Loading

    /// Load CoreML models. Call once at startup.
    func loadModels() throws {
        guard !isLoaded else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Load unified model (backbone + encoder + decoder)
        guard let unifiedURL = Bundle.main.url(forResource: "sam3_3d_unified_int8",
                                                withExtension: "mlmodelc") else {
            throw LocalInferenceError.modelNotFound("sam3_3d_unified_int8")
        }
        unifiedModel = try MLModel(contentsOf: unifiedURL, configuration: config)

        // Load 3D head
        guard let headURL = Bundle.main.url(forResource: "head3d_int8",
                                            withExtension: "mlmodelc") else {
            throw LocalInferenceError.modelNotFound("head3d_int8")
        }
        head3dModel = try MLModel(contentsOf: headURL, configuration: config)

        // Load pre-computed text embeddings
        try loadTextEmbeddings()

        isLoaded = true
        isModelLoaded = true
        print("[LocalInference] Models loaded successfully")
    }

    private func loadTextEmbeddings() throws {
        // Text embeddings are pre-computed on server and stored as binary
        // Shape: (32, N_texts, 256) for text_features, (N_texts, 32) for text_mask
        // These correspond to the text prompts used during export
        guard let url = Bundle.main.url(forResource: "text_embeddings", withExtension: "bin") else {
            throw LocalInferenceError.modelNotFound("text_embeddings")
        }
        // For now, text embeddings will be loaded from the binary file
        // Format: [n_texts: int32, seq_len: int32, dim: int32, features_data: float32[], mask_data: bool[]]
        let data = try Data(contentsOf: url)
        // Parse header
        let n_texts = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Int32.self) }
        let seq_len = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: Int32.self) }
        let dim = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: Int32.self) }

        // Create MLMultiArray for text_features: (seq_len, n_texts, dim)
        textFeatures = try MLMultiArray(shape: [NSNumber(value: seq_len),
                                                 NSNumber(value: n_texts),
                                                 NSNumber(value: dim)],
                                        dataType: .float32)
        let featSize = Int(seq_len * n_texts * dim)
        let featPtr = textFeatures!.dataPointer.bindMemory(to: Float.self, capacity: featSize)
        data.withUnsafeBytes { rawPtr in
            let srcPtr = rawPtr.baseAddress!.advanced(by: 12).assumingMemoryBound(to: Float.self)
            featPtr.update(from: srcPtr, count: featSize)
        }

        // Create MLMultiArray for text_mask: (n_texts, seq_len)
        textMask = try MLMultiArray(shape: [NSNumber(value: n_texts),
                                             NSNumber(value: seq_len)],
                                    dataType: .float32)
        // Mask is all False (no padding) for our pre-computed prompts
        let maskPtr = textMask!.dataPointer.bindMemory(to: Float.self,
                                                        capacity: Int(n_texts * seq_len))
        for i in 0..<Int(n_texts * seq_len) {
            maskPtr[i] = 0.0
        }

        print("[LocalInference] Text embeddings: \(n_texts) texts, seq_len=\(seq_len), dim=\(dim)")
    }

    // MARK: - Detection

    /// Run detection on captured data. Same interface as DetectionService.
    func detect(capture: CaptureData) async throws -> DetectionResponse {
        guard isLoaded, let unifiedModel = unifiedModel else {
            throw LocalInferenceError.notLoaded
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Preprocess image
        let (inputArray, metadata) = try preprocessImage(capture.pngData)

        // 2. Run unified model (backbone + encoder + decoder)
        let unifiedInput = try MLDictionaryFeatureProvider(dictionary: [
            "image": inputArray,
            "text_features": textFeatures!,
            "text_mask": textMask!,
        ])
        let unifiedOutput = try unifiedModel.prediction(from: unifiedInput)

        // Extract outputs
        guard let logits = unifiedOutput.featureValue(for: "pred_logits")?.multiArrayValue,
              let boxesNorm = unifiedOutput.featureValue(for: "pred_boxes_xyxy")?.multiArrayValue,
              let queries = unifiedOutput.featureValue(for: "queries")?.multiArrayValue else {
            throw LocalInferenceError.invalidOutput
        }

        // 3. Decode detections
        let boxes = decodeDetections(
            logits: logits,
            boxesNorm: boxesNorm,
            metadata: metadata,
            cameraToWorld: capture.cameraToWorld,
            intrinsicK: capture.intrinsicK,
            scoreThreshold: scoreThreshold
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("[LocalInference] Detection done: \(boxes.count) objects in \(String(format: "%.1f", elapsed))s")

        return DetectionResponse(boxes: boxes, mode: "local_int8")
    }

    // MARK: - Preprocessing

    private func preprocessImage(_ pngData: Data) throws -> (MLMultiArray, PreprocessMetadata) {
        guard let uiImage = UIImage(data: pngData),
              let cgImage = uiImage.cgImage else {
            throw LocalInferenceError.invalidImage
        }

        let origW = cgImage.width
        let origH = cgImage.height

        // Resize keeping aspect ratio
        let scale = Float(inputSize) / Float(max(origW, origH))
        let newW = Int(Float(origW) * scale)
        let newH = Int(Float(origH) * scale)

        // Center padding
        let padLeft = (inputSize - newW) / 2
        let padTop = (inputSize - newH) / 2

        // Create input array (1, 3, 1008, 1008)
        let inputArray = try MLMultiArray(shape: [1, 3, NSNumber(value: inputSize),
                                                   NSNumber(value: inputSize)],
                                          dataType: .float32)
        let ptr = inputArray.dataPointer.bindMemory(to: Float.self,
                                                     capacity: 3 * inputSize * inputSize)

        // Zero-fill (padding)
        memset(ptr, 0, 3 * inputSize * inputSize * MemoryLayout<Float>.size)

        // Resize image to newW x newH
        let context = CGContext(data: nil, width: newW, height: newH,
                                bitsPerComponent: 8, bytesPerRow: newW * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let pixels = context.data else {
            throw LocalInferenceError.invalidImage
        }
        let pixelPtr = pixels.bindMemory(to: UInt8.self, capacity: newW * newH * 4)

        // Fill with normalized values (CHW format with ImageNet normalization)
        for y in 0..<newH {
            for x in 0..<newW {
                let srcIdx = (y * newW + x) * 4
                let dstY = y + padTop
                let dstX = x + padLeft

                for c in 0..<3 {
                    let pixelVal = Float(pixelPtr[srcIdx + c]) / 255.0
                    let normalized = (pixelVal - mean[c]) / std[c]
                    let dstIdx = c * inputSize * inputSize + dstY * inputSize + dstX
                    ptr[dstIdx] = normalized
                }
            }
        }

        let metadata = PreprocessMetadata(
            originalW: origW, originalH: origH,
            newW: newW, newH: newH,
            padLeft: padLeft, padTop: padTop,
            scale: scale
        )

        return (inputArray, metadata)
    }

    // MARK: - Postprocessing

    private func decodeDetections(
        logits: MLMultiArray,
        boxesNorm: MLMultiArray,
        metadata: PreprocessMetadata,
        cameraToWorld: [[Float]],
        intrinsicK: [[Float]],
        scoreThreshold: Float
    ) -> [BBox3D] {
        let nPrompts = logits.shape[0].intValue
        let nQueries = logits.shape[1].intValue
        let logitsPtr = logits.dataPointer.bindMemory(to: Float.self,
                                                       capacity: nPrompts * nQueries)
        let boxesPtr = boxesNorm.dataPointer.bindMemory(to: Float.self,
                                                         capacity: nPrompts * nQueries * 4)

        let prompts = textPrompt.split(separator: ".").map(String.init)
        var results: [BBox3D] = []

        for p in 0..<nPrompts {
            let label = p < prompts.count ? prompts[p] : "object"

            for q in 0..<nQueries {
                let logit = logitsPtr[p * nQueries + q]
                let score = 1.0 / (1.0 + exp(-logit))  // sigmoid

                if score < scoreThreshold { continue }

                // Decode normalized xyxy to pixel coords
                let bIdx = (p * nQueries + q) * 4
                var x1 = boxesPtr[bIdx + 0] * Float(inputSize)
                var y1 = boxesPtr[bIdx + 1] * Float(inputSize)
                var x2 = boxesPtr[bIdx + 2] * Float(inputSize)
                var y2 = boxesPtr[bIdx + 3] * Float(inputSize)

                // Remove padding
                x1 -= Float(metadata.padLeft)
                y1 -= Float(metadata.padTop)
                x2 -= Float(metadata.padLeft)
                y2 -= Float(metadata.padTop)

                // Scale to original size
                let scaleX = Float(metadata.originalW) / Float(metadata.newW)
                let scaleY = Float(metadata.originalH) / Float(metadata.newH)
                x1 *= scaleX; x2 *= scaleX
                y1 *= scaleY; y2 *= scaleY

                // 2D box center for approximate 3D (without depth model)
                let cx = (x1 + x2) / 2.0
                let cy = (y1 + y2) / 2.0
                let w = x2 - x1
                let h = y2 - y1

                // Approximate 3D center using intrinsics + estimated depth
                // depth = focal_length * typical_size / box_size
                let fx = intrinsicK[0][0]
                let fy = intrinsicK[1][1]
                let ppx = intrinsicK[0][2]
                let ppy = intrinsicK[1][2]

                let typicalSize: Float = 0.5  // rough average object size in meters
                let depth = fx * typicalSize / max(w, h)

                // Unproject to camera coords
                let camX = (cx - ppx) * depth / fx
                let camY = (cy - ppy) * depth / fy
                let camZ = depth

                // Transform to world coords
                let cam2world = cameraToWorld
                let worldX = cam2world[0][0] * camX + cam2world[0][1] * camY + cam2world[0][2] * camZ + cam2world[0][3]
                let worldY = cam2world[1][0] * camX + cam2world[1][1] * camY + cam2world[1][2] * camZ + cam2world[1][3]
                let worldZ = cam2world[2][0] * camX + cam2world[2][1] * camY + cam2world[2][2] * camZ + cam2world[2][3]

                // Estimate size from 2D box
                let sizeW = w * depth / fx
                let sizeH = h * depth / fy
                let sizeD = (sizeW + sizeH) / 2.0

                let color = colorForPrompt(p)

                results.append(BBox3D(
                    label: label,
                    center: [worldX, worldY, worldZ],
                    size: [sizeW, sizeH, sizeD],
                    rotation: [0, 0, 0, 1],  // identity quaternion
                    color: color,
                    score: score,
                    n_frames: nil
                ))
            }
        }

        // Simple NMS by center distance
        results = nmsBy3DDistance(results, threshold: 0.3)

        return results
    }

    private func nmsBy3DDistance(_ boxes: [BBox3D], threshold: Float) -> [BBox3D] {
        let sorted = boxes.sorted { $0.score > $1.score }
        var kept: [BBox3D] = []

        for box in sorted {
            let dominated = kept.contains { existing in
                let dx = existing.center[0] - box.center[0]
                let dy = existing.center[1] - box.center[1]
                let dz = existing.center[2] - box.center[2]
                let dist = sqrt(dx*dx + dy*dy + dz*dz)
                return dist < threshold
            }
            if !dominated {
                kept.append(box)
            }
        }
        return kept
    }

    private func colorForPrompt(_ idx: Int) -> [Float] {
        let colors: [[Float]] = [
            [0.0, 1.0, 0.0],  // green
            [0.0, 0.0, 1.0],  // blue
            [1.0, 0.0, 0.0],  // red
            [1.0, 1.0, 0.0],  // yellow
            [1.0, 0.0, 1.0],  // magenta
        ]
        return colors[idx % colors.count]
    }
}

// MARK: - Helper Types

private struct PreprocessMetadata {
    let originalW: Int
    let originalH: Int
    let newW: Int
    let newH: Int
    let padLeft: Int
    let padTop: Int
    let scale: Float
}

enum LocalInferenceError: LocalizedError {
    case modelNotFound(String)
    case notLoaded
    case invalidImage
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name): return "CoreML model not found: \(name)"
        case .notLoaded: return "Models not loaded. Call loadModels() first."
        case .invalidImage: return "Failed to process image"
        case .invalidOutput: return "Invalid model output"
        }
    }
}
