import Foundation
import Accelerate
import simd
import onnxruntime_objc

/// On-device SAM3_3D inference using ONNX Runtime.
/// Drop-in replacement for DetectionService - same CaptureData input, same DetectionResponse output.
///
/// Model file: sam3_3d_raw.onnx (~3GB, downloaded on first use)
/// Inputs: image (1,3,1008,1008) + depth (1,1,1008,1008) + intrinsics (1,3,3)
/// Outputs: raw logits + boxes2d + boxes3d + conf3d + depth_map + K_pred
class LocalInferenceService {
    static let shared = LocalInferenceService()

    private var session: ORTSession?
    private var env: ORTEnv?
    private(set) var isModelLoaded = false
    private var tokenizer: BPETokenizer?

    // Model runs 1 prompt at a time, Swift loops over categories
    private let nQueries = 200
    private let contextLength = 32  // text encoder context length

    // SAM3 input size
    private let inputSize = 1008

    // ImageNet normalization
    private let imgMean: [Float] = [0.485, 0.456, 0.406]
    private let imgStd: [Float] = [0.229, 0.224, 0.225]

    // Box coder constants (from coder.py)
    private let centerScale: Float = 10.0
    private let depthScale: Float = 2.0
    private let dimScale: Float = 2.0

    // NMS
    private let iouThreshold: Float = 0.6
    private let nmsDistance: Float = 0.3  // cross-class NMS distance in meters

    // Category colors (same as API server)
    private let categoryColors: [[Float]] = [
        [0.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.5], [0.0, 1.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 0.0],
    ]

    private init() {}

    // MARK: - Model Loading

    func loadModels() throws {
        guard !isModelLoaded else { return }

        env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        // CPU execution with INT8 — no dequantization, low memory usage
        // ARM NEON provides hardware INT8 acceleration

        guard let modelPath = getModelPath() else {
            throw LocalInferenceError.modelNotFound("sam3_3d_raw.onnx")
        }

        session = try ORTSession(env: env!, modelPath: modelPath, sessionOptions: options)

        // Load BPE tokenizer
        guard let vocabPath = getVocabPath() else {
            throw LocalInferenceError.modelNotFound("bpe_vocab.txt")
        }
        tokenizer = try BPETokenizer(vocabPath: vocabPath)

        isModelLoaded = true
        print("[LocalInference] Model + tokenizer loaded")
    }

    // HuggingFace URLs
    private static let modelURL = "https://huggingface.co/weikaih/iphone_test/resolve/main/model_int8.onnx"
    private static let vocabURL = "https://huggingface.co/weikaih/iphone_test/resolve/main/bpe_vocab.txt"

    private func getModelPath() -> String? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelPath = docs.appendingPathComponent("sam3_3d_model.onnx").path
        if FileManager.default.fileExists(atPath: modelPath) {
            return modelPath
        }
        return Bundle.main.path(forResource: "model", ofType: "onnx")
    }

    private func getVocabPath() -> String? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let vocabPath = docs.appendingPathComponent("bpe_vocab.txt").path
        if FileManager.default.fileExists(atPath: vocabPath) {
            return vocabPath
        }
        return Bundle.main.path(forResource: "bpe_vocab", ofType: "txt")
    }

    /// Download model + vocab from HuggingFace if not present.
    func downloadModelIfNeeded(progress: @escaping (String) -> Void) async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        let modelPath = docs.appendingPathComponent("sam3_3d_model.onnx")
        if !FileManager.default.fileExists(atPath: modelPath.path) {
            progress("Downloading model (1.1GB)...")
            let (tempURL, _) = try await URLSession.shared.download(from: URL(string: Self.modelURL)!)
            try FileManager.default.moveItem(at: tempURL, to: modelPath)
        }

        let vocabPath = docs.appendingPathComponent("bpe_vocab.txt")
        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            progress("Downloading tokenizer vocab...")
            let (tempURL, _) = try await URLSession.shared.download(from: URL(string: Self.vocabURL)!)
            try FileManager.default.moveItem(at: tempURL, to: vocabPath)
        }

        progress("Model ready!")
    }

    /// Download a file with progress reporting.
    private func downloadFile(from urlString: String, to destination: URL, onProgress: @escaping (Int) -> Void) async throws {
        guard let url = URL(string: urlString) else {
            throw LocalInferenceError.modelNotFound("Invalid URL: \(urlString)")
        }

        let delegate = DownloadProgressDelegate(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        let (tempURL, response) = try await session.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw LocalInferenceError.modelNotFound("Download failed (HTTP \(code)): \(urlString)")
        }

        // Remove existing file if present (e.g., partial download from before)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    // MARK: - Detection (same interface as DetectionService)

    func detect(capture: CaptureData) async throws -> DetectionResponse {
        guard isModelLoaded, let session = session, let tokenizer = tokenizer else {
            throw LocalInferenceError.notLoaded
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Preprocess image + depth + intrinsics (same for all categories)
        let preprocessed = try preprocess(capture: capture)

        // 2. Parse categories from text prompt
        let textPrompt = DetectionService.shared.textPrompt
        let categories = textPrompt.split(separator: ".").map { String($0).trimmingCharacters(in: .whitespaces) }
        let scoreThreshold = DetectionService.shared.scoreThreshold

        // 3. Loop over categories — 1 prompt per ONNX call
        var allProposals: [(score2d: Float, scoreAll: Float, box2d: [Float], box3d: [Float], classId: Int)] = []

        for (classIdx, category) in categories.enumerated() {
            // Tokenize
            let tokenIds = tokenizer.encode(category, contextLength: contextLength)
            var tokenIdsInt64 = tokenIds.map { Int64($0) }

            // Create token_ids ORTValue (1, 32) int64
            let tokenData = NSMutableData(bytes: &tokenIdsInt64,
                                          length: contextLength * MemoryLayout<Int64>.size)
            let tokenValue = try ORTValue(tensorData: tokenData, elementType: .int64,
                                          shape: [1, NSNumber(value: contextLength)])

            // Run inference for this single category
            let outputs = try runInference(session: session, input: preprocessed, tokenIds: tokenValue)

            // Extract proposals for this category (N=1 prompt, S=200 queries)
            let S = nQueries
            let H = Float(inputSize)
            let W = Float(inputSize)

            let presenceScore = sigmoid(outputs.presenceLogits[0])

            for q in 0..<S {
                let logit2d = outputs.predLogits[q]
                let logit3d = outputs.predConf3d[q]
                let score2d = sigmoid(logit2d) * presenceScore
                let score3d = sigmoid(logit3d)
                let scoreAll = score2d + 0.5 * score3d

                if score2d < scoreThreshold { continue }

                let b2dBase = q * 4
                let box2d = [
                    outputs.predBoxes2d[b2dBase] * W,
                    outputs.predBoxes2d[b2dBase + 1] * H,
                    outputs.predBoxes2d[b2dBase + 2] * W,
                    outputs.predBoxes2d[b2dBase + 3] * H,
                ]
                let b3dBase = q * 12
                let box3d = Array(outputs.predBoxes3d[b3dBase..<b3dBase + 12])

                allProposals.append((score2d, scoreAll, box2d, box3d, classIdx))
            }
        }

        if allProposals.isEmpty {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("[LocalInference] 0 detections in \(String(format: "%.1f", elapsed))s")
            return DetectionResponse(boxes: [], mode: "local_onnx", predicted_intrinsics: nil)
        }

        // 4. Per-class NMS on merged proposals
        let kept = batchedNMS(proposals: allProposals, iouThreshold: iouThreshold)

        // 5. Decode 3D boxes + camera-to-world
        let K = preprocessed.intrinsicsPadded
        var worldBoxes: [BBox3D] = []

        for idx in kept {
            let prop = allProposals[idx]
            let decoded = decodeBox3D(box2d: prop.box2d, box3dEncoded: prop.box3d, K: K)
            let worldResult = cameraToWorldTransform(
                center: [decoded.cx, decoded.cy, decoded.cz],
                dims: [decoded.w, decoded.l, decoded.h],
                quaternion: decoded.quaternion,
                cameraToWorld: capture.cameraToWorld
            )

            let classIdx = prop.classId
            let label = classIdx < categories.count ? categories[classIdx] : "object"
            let colorIdx = classIdx % categoryColors.count

            worldBoxes.append(BBox3D(
                label: label,
                center: worldResult.center,
                size: worldResult.size,
                rotation: worldResult.rotation,
                color: categoryColors[colorIdx],
                score: prop.score2d,
                n_frames: nil,
                projected_corners: nil
            ))
        }

        // 6. Cross-class NMS
        if categories.count > 1 && worldBoxes.count > 1 {
            worldBoxes = crossClassNMS(worldBoxes, distance: nmsDistance)
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("[LocalInference] \(worldBoxes.count) detections in \(String(format: "%.1f", elapsed))s (\(categories.count) categories)")

        return DetectionResponse(boxes: worldBoxes, mode: "local_onnx", predicted_intrinsics: nil)
    }

    // MARK: - Preprocessing

    private struct PreprocessedInput {
        let imageArray: ORTValue
        let depthArray: ORTValue
        let intrinsicsArray: ORTValue
        let originalW: Int
        let originalH: Int
        let newW: Int
        let newH: Int
        let padLeft: Int
        let padTop: Int
        let padRight: Int
        let padBottom: Int
        let intrinsicsPadded: [[Float]]  // (3,3) padded-space K for 3D decoding
    }

    private func preprocess(capture: CaptureData) throws -> PreprocessedInput {
        // Decode PNG -> RGB pixels
        guard let uiImage = UIImage(data: capture.pngData),
              let cgImage = uiImage.cgImage else {
            throw LocalInferenceError.invalidImage
        }
        let origW = cgImage.width
        let origH = cgImage.height

        // Step 1: fix_cy (same as API fix_intrinsics_cy, source="iphone")
        // iPhone sends cy in bottom-left convention, convert to OpenCV top-left
        var K = capture.intrinsicK  // [[Float]] 3x3
        K[1][2] = Float(origH) - K[1][2]

        // Step 2: Resize to fit 1008x1008 preserving aspect ratio
        let scale = Float(inputSize) / Float(max(origW, origH))
        let newW = Int(Float(origW) * scale)
        let newH = Int(Float(origH) * scale)

        // Step 3: Scale intrinsics by resize ratio
        K[0][0] *= scale; K[0][2] *= scale  // fx, cx
        K[1][1] *= scale; K[1][2] *= scale  // fy, cy

        // Step 4: Center pad
        let padLeft = (inputSize - newW) / 2
        let padTop = (inputSize - newH) / 2
        let padRight = inputSize - newW - padLeft
        let padBottom = inputSize - newH - padTop

        // Step 5: Adjust intrinsics for padding
        K[0][2] += Float(padLeft)   // cx += pad_left
        K[1][2] += Float(padTop)    // cy += pad_top

        // Step 6: Create image tensor (1, 3, 1008, 1008) with ImageNet normalization
        let imageData = try createImageTensor(cgImage: cgImage, newW: newW, newH: newH,
                                               padLeft: padLeft, padTop: padTop)

        // Step 7: Create depth tensor (1, 1, 1008, 1008)
        let depthData = try createDepthTensor(depthPng: capture.depthPngData,
                                               origW: origW, origH: origH,
                                               newW: newW, newH: newH,
                                               padLeft: padLeft, padTop: padTop)

        // Step 8: Create intrinsics tensor (1, 3, 3)
        var kFlat: [Float] = []
        for row in K { kFlat.append(contentsOf: row) }
        let intrinsicsData = try ORTValue(
            tensorData: NSMutableData(bytes: &kFlat, length: 9 * MemoryLayout<Float>.size),
            elementType: .float,
            shape: [1, 3, 3]
        )

        return PreprocessedInput(
            imageArray: imageData, depthArray: depthData, intrinsicsArray: intrinsicsData,
            originalW: origW, originalH: origH, newW: newW, newH: newH,
            padLeft: padLeft, padTop: padTop, padRight: padRight, padBottom: padBottom,
            intrinsicsPadded: K
        )
    }

    private func createImageTensor(cgImage: CGImage, newW: Int, newH: Int,
                                    padLeft: Int, padTop: Int) throws -> ORTValue {
        // Resize image
        let context = CGContext(data: nil, width: newW, height: newH,
                                bitsPerComponent: 8, bytesPerRow: newW * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let pixels = context.data else { throw LocalInferenceError.invalidImage }
        let pixelPtr = pixels.bindMemory(to: UInt8.self, capacity: newW * newH * 4)

        // Create CHW tensor with ImageNet normalization + center padding
        // CRITICAL: padding area must be normalized too (same as Python NormalizeImages)
        // Python normalizes the whole 1008x1008 including padding (black=0) -> (-mean/std)
        let totalSize = 3 * inputSize * inputSize
        var buffer = [Float](repeating: 0.0, count: totalSize)
        // Fill with normalized zero (what ImageNet norm gives for black pixels)
        for c in 0..<3 {
            let normZero = (0.0 - imgMean[c]) / imgStd[c]
            let offset = c * inputSize * inputSize
            for i in 0..<(inputSize * inputSize) {
                buffer[offset + i] = normZero
            }
        }

        for y in 0..<newH {
            for x in 0..<newW {
                let srcIdx = (y * newW + x) * 4
                let dstY = y + padTop
                let dstX = x + padLeft
                for c in 0..<3 {
                    let pixelVal = Float(pixelPtr[srcIdx + c]) / 255.0
                    let normalized = (pixelVal - imgMean[c]) / imgStd[c]
                    buffer[c * inputSize * inputSize + dstY * inputSize + dstX] = normalized
                }
            }
        }

        let data = NSMutableData(bytes: &buffer, length: totalSize * MemoryLayout<Float>.size)
        return try ORTValue(tensorData: data, elementType: .float,
                           shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)])
    }

    private func createDepthTensor(depthPng: Data?, origW: Int, origH: Int,
                                    newW: Int, newH: Int,
                                    padLeft: Int, padTop: Int) throws -> ORTValue {
        let totalSize = inputSize * inputSize
        var buffer = [Float](repeating: 0.0, count: totalSize)

        if let depthData = depthPng {
            // Decode 16-bit PNG -> meters (same as API preprocess_depth)
            // API: depth_mm = cv2.imdecode(data, IMREAD_UNCHANGED)
            //      depth_m = depth_mm.astype(float32) / 1000.0
            //      resize to (newW, newH) NEAREST
            //      center pad
            // For simplicity, decode depth from the raw 16-bit PNG data
            // iPhone encodes depth as 16-bit grayscale PNG in millimeters
            if let depthImage = UIImage(data: depthData)?.cgImage {
                let dW = depthImage.width
                let dH = depthImage.height

                // Get raw pixel data (16-bit)
                let depthContext = CGContext(data: nil, width: dW, height: dH,
                                            bitsPerComponent: 16, bytesPerRow: dW * 2,
                                            space: CGColorSpaceCreateDeviceGray(),
                                            bitmapInfo: CGImageAlphaInfo.none.rawValue)
                depthContext?.draw(depthImage, in: CGRect(x: 0, y: 0, width: dW, height: dH))

                if let depthPixels = depthContext?.data {
                    let uint16Ptr = depthPixels.bindMemory(to: UInt16.self, capacity: dW * dH)

                    // Resize (nearest neighbor) + pad
                    let scaleX = Float(dW) / Float(newW)
                    let scaleY = Float(dH) / Float(newH)

                    for y in 0..<newH {
                        for x in 0..<newW {
                            let srcX = min(Int(Float(x) * scaleX), dW - 1)
                            let srcY = min(Int(Float(y) * scaleY), dH - 1)
                            let mm = Float(uint16Ptr[srcY * dW + srcX])
                            let meters = mm / 1000.0

                            let dstY = y + padTop
                            let dstX = x + padLeft
                            buffer[dstY * inputSize + dstX] = meters
                        }
                    }
                }
            }
        }

        let data = NSMutableData(bytes: &buffer, length: totalSize * MemoryLayout<Float>.size)
        return try ORTValue(tensorData: data, elementType: .float,
                           shape: [1, 1, NSNumber(value: inputSize), NSNumber(value: inputSize)])
    }

    // MARK: - Inference

    private struct ModelOutputs {
        let predLogits: [Float]      // (1, 200, 1) flattened = 200 values
        let predBoxes2d: [Float]     // (1, 200, 4) flattened = 800 values
        let predBoxes3d: [Float]     // (1, 200, 12) flattened = 2400 values
        let predConf3d: [Float]      // (1, 200, 1) flattened = 200 values
        let presenceLogits: [Float]  // (1, 1) flattened = 1 value
    }

    private func runInference(session: ORTSession, input: PreprocessedInput, tokenIds: ORTValue) throws -> ModelOutputs {
        let outputs = try session.run(
            withInputs: [
                "image": input.imageArray,
                "depth": input.depthArray,
                "intrinsics": input.intrinsicsArray,
                "token_ids": tokenIds,
            ],
            outputNames: ["pred_logits", "pred_boxes_2d", "pred_boxes_3d", "pred_conf_3d", "presence_logits"],
            runOptions: nil
        )

        func extractFloats(_ value: ORTValue) throws -> [Float] {
            let data = try value.tensorData() as Data
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        return ModelOutputs(
            predLogits: try extractFloats(outputs["pred_logits"]!),
            predBoxes2d: try extractFloats(outputs["pred_boxes_2d"]!),
            predBoxes3d: try extractFloats(outputs["pred_boxes_3d"]!),
            predConf3d: try extractFloats(outputs["pred_conf_3d"]!),
            presenceLogits: try extractFloats(outputs["presence_logits"]!)
        )
    }

    // MARK: - 3D Box Decoding

    private struct DecodedBox3D {
        let cx: Float, cy: Float, cz: Float
        let w: Float, l: Float, h: Float
        let quaternion: [Float]  // [qw, qx, qy, qz]
    }

    private func decodeBox3D(box2d: [Float], box3dEncoded: [Float], K: [[Float]]) -> DecodedBox3D {
        // box2d: pixel xyxy in padded space
        // box3dEncoded: [delta_cx, delta_cy, log_depth, log_w, log_l, log_h, r6d_0..5]

        // 2D box center
        let ctrX = (box2d[0] + box2d[2]) / 2.0
        let ctrY = (box2d[1] + box2d[3]) / 2.0

        // Delta center (scaled)
        let deltaCx = box3dEncoded[0] * centerScale
        let deltaCy = box3dEncoded[1] * centerScale

        // Projected 3D center
        let projCx = ctrX + deltaCx
        let projCy = ctrY + deltaCy

        // Depth
        let depth = exp(box3dEncoded[2] / depthScale)

        // Unproject to camera 3D (using padded-space K)
        let fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2]
        let camX = (projCx - cx) * depth / fx
        let camY = (projCy - cy) * depth / fy
        let camZ = depth

        // Dimensions
        let dimW = exp(box3dEncoded[3] / dimScale)
        let dimL = exp(box3dEncoded[4] / dimScale)
        let dimH = exp(box3dEncoded[5] / dimScale)

        // Rotation: 6D -> matrix -> quaternion + canonical normalization
        let rot6d = Array(box3dEncoded[6..<12])
        var rotMatrix = rotation6dToMatrix(rot6d)
        var finalW = dimW, finalL = dimL

        // Canonical rotation normalization (from coder.py _normalize_canonical)
        // Step 1: Force W <= L. If W > L, swap dims and apply Ry(90)
        if finalW > finalL {
            let tmp = finalW; finalW = finalL; finalL = tmp
            // Ry(90): new col0 = -old col2, new col2 = old col0
            let col0 = [rotMatrix[0][0], rotMatrix[1][0], rotMatrix[2][0]]
            let col2 = [rotMatrix[0][2], rotMatrix[1][2], rotMatrix[2][2]]
            rotMatrix[0][0] = -col2[0]; rotMatrix[1][0] = -col2[1]; rotMatrix[2][0] = -col2[2]
            rotMatrix[0][2] = col0[0]; rotMatrix[1][2] = col0[1]; rotMatrix[2][2] = col0[2]
        }

        // Step 2: Normalize yaw to [0, pi). yaw = atan2(-R[2,0], R[0,0])
        let yaw = atan2(-rotMatrix[2][0], rotMatrix[0][0])
        if yaw < 0 || yaw > Float.pi - 1e-4 {
            // Apply Ry(180): negate columns 0 and 2
            for r in 0..<3 {
                rotMatrix[r][0] = -rotMatrix[r][0]
                rotMatrix[r][2] = -rotMatrix[r][2]
            }
        }

        let quat = matrixToQuaternion(rotMatrix)  // [qw, qx, qy, qz]

        return DecodedBox3D(cx: camX, cy: camY, cz: camZ,
                           w: finalW, l: finalL, h: dimH, quaternion: quat)
    }

    // MARK: - Camera-to-World Transform

    private struct WorldBox {
        let center: [Float]    // [x, y, z] world coords
        let size: [Float]      // [w, h, d] world dims
        let rotation: [Float]  // [qx, qy, qz, qw] (scipy convention)
    }

    private func cameraToWorldTransform(
        center: [Float], dims: [Float], quaternion: [Float],
        cameraToWorld: [[Float]]
    ) -> WorldBox {
        // Y-flip for iPhone (OpenCV Y-down -> ARKit Y-up)
        // T = camera_to_world @ diag(1, -1, 1, 1)
        let flipY4x4: [[Float]] = [
            [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ]
        let T = matMul4x4(cameraToWorld, flipY4x4)

        // Transform center
        let centerH: [Float] = [center[0], center[1], center[2], 1.0]
        let centerWorld = matVecMul4x4(T, centerH)

        // Transform rotation
        let R_cam2world = extract3x3(from: cameraToWorld)
        let flipY3x3: [[Float]] = [[1,0,0], [0,-1,0], [0,0,1]]
        let R_obj = quaternionToMatrix(quaternion)  // quat is [qw, qx, qy, qz]

        // R_world = R_cam2world @ flip_y @ R_obj @ flip_y
        let R_world = matMul3x3(matMul3x3(matMul3x3(R_cam2world, flipY3x3), R_obj), flipY3x3)
        let quatWorld = matrixToQuaternion(R_world)

        // Dimension permutation: [w, l, h] -> [w, h, d] (same as API line 316)
        let sizeWorld = [dims[0], dims[2], dims[1]]

        // Quaternion: convert from [qw, qx, qy, qz] to [qx, qy, qz, qw] (API convention)
        let rotOut = [quatWorld[1], quatWorld[2], quatWorld[3], quatWorld[0]]

        return WorldBox(
            center: Array(centerWorld[0..<3]),
            size: sizeWorld,
            rotation: rotOut
        )
    }

    // MARK: - Math Utilities

    private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }

    private func normalize(_ v: [Float]) -> [Float] {
        let len = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        guard len > 1e-8 else { return [1, 0, 0] }
        return [v[0]/len, v[1]/len, v[2]/len]
    }

    private func cross(_ a: [Float], _ b: [Float]) -> [Float] {
        [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
    }

    private func dot3(_ a: [Float], _ b: [Float]) -> Float {
        a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    }

    /// 6D rotation representation -> 3x3 matrix (Gram-Schmidt)
    /// From vis4d/op/geometry/rotation.py rotation_6d_to_matrix
    private func rotation6dToMatrix(_ d6: [Float]) -> [[Float]] {
        let a1 = Array(d6[0..<3])
        let a2 = Array(d6[3..<6])
        let b1 = normalize(a1)
        let d = dot3(b1, a2)
        let b2 = normalize([a2[0] - d*b1[0], a2[1] - d*b1[1], a2[2] - d*b1[2]])
        let b3 = cross(b1, b2)
        // Row-major: row0=b1, row1=b2, row2=b3
        return [b1, b2, b3]
    }

    /// 3x3 rotation matrix -> quaternion [qw, qx, qy, qz] (Shepperd method)
    /// From vis4d/op/geometry/rotation.py matrix_to_quaternion
    private func matrixToQuaternion(_ m: [[Float]]) -> [Float] {
        let m00 = m[0][0], m01 = m[0][1], m02 = m[0][2]
        let m10 = m[1][0], m11 = m[1][1], m12 = m[1][2]
        let m20 = m[2][0], m21 = m[2][1], m22 = m[2][2]

        let qAbs: [Float] = [
            sqrt(max(0, 1.0 + m00 + m11 + m22)) / 2.0,
            sqrt(max(0, 1.0 + m00 - m11 - m22)) / 2.0,
            sqrt(max(0, 1.0 - m00 + m11 - m22)) / 2.0,
            sqrt(max(0, 1.0 - m00 - m11 + m22)) / 2.0,
        ]

        // Pick best-conditioned
        let bestIdx = qAbs.enumerated().max(by: { $0.element < $1.element })!.offset

        var qw: Float = 0, qx: Float = 0, qy: Float = 0, qz: Float = 0
        switch bestIdx {
        case 0:
            qw = qAbs[0]
            qx = (m21 - m12) / (4.0 * qw)
            qy = (m02 - m20) / (4.0 * qw)
            qz = (m10 - m01) / (4.0 * qw)
        case 1:
            qx = qAbs[1]
            qw = (m21 - m12) / (4.0 * qx)
            qy = (m01 + m10) / (4.0 * qx)
            qz = (m02 + m20) / (4.0 * qx)
        case 2:
            qy = qAbs[2]
            qw = (m02 - m20) / (4.0 * qy)
            qx = (m01 + m10) / (4.0 * qy)
            qz = (m12 + m21) / (4.0 * qy)
        default:
            qz = qAbs[3]
            qw = (m10 - m01) / (4.0 * qz)
            qx = (m02 + m20) / (4.0 * qz)
            qy = (m12 + m21) / (4.0 * qz)
        }

        return [qw, qx, qy, qz]
    }

    /// Quaternion [qw, qx, qy, qz] -> 3x3 rotation matrix
    private func quaternionToMatrix(_ q: [Float]) -> [[Float]] {
        let w = q[0], x = q[1], y = q[2], z = q[3]
        return [
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ]
    }

    private func matMul3x3(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
        var r = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
        for i in 0..<3 { for j in 0..<3 { for k in 0..<3 {
            r[i][j] += a[i][k] * b[k][j]
        }}}
        return r
    }

    private func matMul4x4(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
        var r = [[Float]](repeating: [Float](repeating: 0, count: 4), count: 4)
        for i in 0..<4 { for j in 0..<4 { for k in 0..<4 {
            r[i][j] += a[i][k] * b[k][j]
        }}}
        return r
    }

    private func matVecMul4x4(_ m: [[Float]], _ v: [Float]) -> [Float] {
        var r = [Float](repeating: 0, count: 4)
        for i in 0..<4 { for j in 0..<4 { r[i] += m[i][j] * v[j] } }
        return r
    }

    private func extract3x3(from m4x4: [[Float]]) -> [[Float]] {
        [[m4x4[0][0], m4x4[0][1], m4x4[0][2]],
         [m4x4[1][0], m4x4[1][1], m4x4[1][2]],
         [m4x4[2][0], m4x4[2][1], m4x4[2][2]]]
    }

    // MARK: - NMS

    private func batchedNMS(proposals: [(score2d: Float, scoreAll: Float, box2d: [Float], box3d: [Float], classId: Int)],
                            iouThreshold: Float) -> [Int] {
        // Per-class NMS (same as torchvision.ops.batched_nms)
        var kept: [Int] = []
        let classes = Set(proposals.map { $0.classId })

        for cls in classes {
            let classIndices = proposals.enumerated()
                .filter { $0.element.classId == cls }
                .map { $0.offset }
                .sorted { proposals[$0].scoreAll > proposals[$1].scoreAll }

            var suppressed = Set<Int>()
            for i in classIndices {
                if suppressed.contains(i) { continue }
                kept.append(i)
                for j in classIndices {
                    if suppressed.contains(j) || j == i { continue }
                    let iou = computeIoU(proposals[i].box2d, proposals[j].box2d)
                    if iou > iouThreshold { suppressed.insert(j) }
                }
            }
        }

        return kept.sorted { proposals[$0].scoreAll > proposals[$1].scoreAll }
    }

    private func computeIoU(_ a: [Float], _ b: [Float]) -> Float {
        let x1 = max(a[0], b[0]), y1 = max(a[1], b[1])
        let x2 = min(a[2], b[2]), y2 = min(a[3], b[3])
        let inter = max(0, x2 - x1) * max(0, y2 - y1)
        let areaA = (a[2] - a[0]) * (a[3] - a[1])
        let areaB = (b[2] - b[0]) * (b[3] - b[1])
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }

    private func crossClassNMS(_ boxes: [BBox3D], distance: Float) -> [BBox3D] {
        let sorted = boxes.sorted { $0.score > $1.score }
        var kept: [BBox3D] = []
        for box in sorted {
            let dominated = kept.contains { existing in
                let dx = existing.center[0] - box.center[0]
                let dy = existing.center[1] - box.center[1]
                let dz = existing.center[2] - box.center[2]
                return sqrt(dx*dx + dy*dy + dz*dz) < distance
            }
            if !dominated { kept.append(box) }
        }
        return kept
    }
}

// MARK: - Errors

enum LocalInferenceError: LocalizedError {
    case modelNotFound(String)
    case notLoaded
    case invalidImage

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name): return "Model not found: \(name)"
        case .notLoaded: return "Models not loaded"
        case .invalidImage: return "Failed to process image"
        }
    }
}

// MARK: - Download Progress Delegate

private class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate {
    let onProgress: (Int) -> Void
    private var lastReported = -1

    init(onProgress: @escaping (Int) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let pct = Int(totalBytesWritten * 100 / totalBytesExpectedToWrite)
        if pct != lastReported {
            lastReported = pct
            onProgress(pct)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // Required delegate method — actual file move handled in downloadFile()
    }
}
