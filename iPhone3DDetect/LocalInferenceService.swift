import Foundation
import Accelerate
import simd
import onnxruntime_objc

/// On-device SAM3_3D inference using ONNX Runtime (3-way split).
/// Drop-in replacement for DetectionService.
///
/// 3 ONNX models loaded one at a time to fit iPhone 3GB memory:
///   1. text_encoder_int8.onnx (339MB): token_ids -> text_features
///   2. depth_model_int4.onnx (243MB): image+depth+intrinsics -> depth_latents
///   3. vision_decoder_int4.onnx (316MB): image+depth_latents+text_features -> detections
///
/// Peak memory = max(339, 243, 316) loaded + cached features (~10MB)
class LocalInferenceService {
    static let shared = LocalInferenceService()

    private(set) var isModelLoaded = false
    private var tokenizer: BPETokenizer?

    private let nQueries = 200
    private let contextLength = 32
    private let textFeatureDim = 256

    private let inputSize = 1008

    private let imgMean: [Float] = [0.485, 0.456, 0.406]
    private let imgStd: [Float] = [0.229, 0.224, 0.225]

    private let centerScale: Float = 10.0
    private let depthScale: Float = 2.0
    private let dimScale: Float = 2.0

    private let iouThreshold: Float = 0.6
    private let nmsDistance: Float = 0.3

    private let categoryColors: [[Float]] = [
        [0.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.5], [0.0, 1.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 0.0],
    ]

    private let textEncoderFile = "text_encoder_int8.onnx"
    private let depthModelFile = "depth_model_int4.onnx"
    private let visionDecoderFile = "vision_decoder_int4.onnx"
    private let vocabFile = "bpe_vocab.txt"

    private init() {}

    // MARK: - Model Loading

    func loadModels() throws {
        guard !isModelLoaded else { return }
        guard let vocabPath = getFilePath(vocabFile) else {
            throw LocalInferenceError.modelNotFound(vocabFile)
        }
        tokenizer = try BPETokenizer(vocabPath: vocabPath)
        for f in [textEncoderFile, depthModelFile, visionDecoderFile] {
            guard getFilePath(f) != nil else { throw LocalInferenceError.modelNotFound(f) }
        }
        isModelLoaded = true
        print("[LocalInference] Tokenizer loaded, 3 models ready")
    }

    private func createIsolatedSession(filename: String) throws -> (ORTEnv, ORTSession) {
        guard let path = getFilePath(filename) else { throw LocalInferenceError.modelNotFound(filename) }
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.basic)
        let session = try ORTSession(env: env, modelPath: path, sessionOptions: options)
        return (env, session)
    }

    private static let baseURL = "https://huggingface.co/weikaih/iphone_test/resolve/main"

    private func getFilePath(_ filename: String) -> String? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let path = docs.appendingPathComponent(filename).path
        if FileManager.default.fileExists(atPath: path) { return path }
        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        return Bundle.main.path(forResource: name, ofType: ext)
    }

    func downloadModelIfNeeded(progress: @escaping (String) -> Void) async throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let files: [(String, String)] = [
            (textEncoderFile, "Downloading text encoder (339MB)..."),
            (depthModelFile, "Downloading depth model (243MB)..."),
            (visionDecoderFile, "Downloading vision decoder (316MB)..."),
            (vocabFile, "Downloading tokenizer vocab..."),
        ]
        for (filename, msg) in files {
            let dest = docs.appendingPathComponent(filename)
            if !FileManager.default.fileExists(atPath: dest.path) {
                progress(msg)
                let url = URL(string: "\(Self.baseURL)/\(filename)")!
                let (tempURL, _) = try await URLSession.shared.download(from: url)
                try FileManager.default.moveItem(at: tempURL, to: dest)
            }
        }
        progress("Models ready!")
    }

    // MARK: - Detection

    func detect(capture: CaptureData) async throws -> DetectionResponse {
        guard isModelLoaded, let tokenizer = tokenizer else { throw LocalInferenceError.notLoaded }
        let startTime = CFAbsoluteTimeGetCurrent()
        let textPrompt = DetectionService.shared.textPrompt
        let categories = textPrompt.split(separator: ".").map { String($0).trimmingCharacters(in: .whitespaces) }
        let scoreThreshold = DetectionService.shared.scoreThreshold

        // Phase 1: Text encoding (339MB)
        print("[LocalInference] Phase 1: text encoding (\(categories.count) cats)...")
        var categoryFeatures: [(features: [Float], mask: [Int64])] = []
        try autoreleasepool {
            let (env, session) = try createIsolatedSession(filename: textEncoderFile)
            _ = env
            for category in categories {
                let tokenIds = tokenizer.encode(category, contextLength: contextLength)
                var ids = tokenIds.map { Int64($0) }
                let idData = NSMutableData(bytes: &ids, length: contextLength * MemoryLayout<Int64>.size)
                let idValue = try ORTValue(tensorData: idData, elementType: .int64, shape: [1, NSNumber(value: contextLength)])
                let out = try session.run(withInputs: ["token_ids": idValue], outputNames: ["text_features", "text_mask"], runOptions: nil)
                let f = try out["text_features"]!.tensorData() as Data
                let m = try out["text_mask"]!.tensorData() as Data
                categoryFeatures.append((f.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) },
                                         m.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }))
            }
        }
        print("[LocalInference] Phase 1 done: \(String(format: "%.1f", CFAbsoluteTimeGetCurrent()-startTime))s")

        // Phase 2: Depth (243MB)
        print("[LocalInference] Phase 2: depth...")
        let preprocessed = try preprocess(capture: capture)
        var depthLatents: [Float] = []
        var rayIntrinsics: [Float] = []
        try autoreleasepool {
            let (env, session) = try createIsolatedSession(filename: depthModelFile)
            _ = env
            let out = try session.run(
                withInputs: ["image": preprocessed.imageArray, "depth": preprocessed.depthArray, "intrinsics": preprocessed.intrinsicsArray],
                outputNames: ["depth_latents", "ray_intrinsics", "depth_map", "K_pred"], runOptions: nil)
            depthLatents = (try out["depth_latents"]!.tensorData() as Data).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            rayIntrinsics = (try out["ray_intrinsics"]!.tensorData() as Data).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        print("[LocalInference] Phase 2 done: \(String(format: "%.1f", CFAbsoluteTimeGetCurrent()-startTime))s")

        // Phase 3: Vision + Decoder (316MB)
        print("[LocalInference] Phase 3: vision+decoder...")
        var allProposals: [(score2d: Float, scoreAll: Float, box2d: [Float], box3d: [Float], classId: Int)] = []
        try autoreleasepool {
            let (env, session) = try createIsolatedSession(filename: visionDecoderFile)
            _ = env
            var dlBuf = depthLatents
            let dlVal = try ORTValue(tensorData: NSMutableData(bytes: &dlBuf, length: dlBuf.count*4), elementType: .float, shape: [1, 2401, 256])
            var riBuf = rayIntrinsics
            let riVal = try ORTValue(tensorData: NSMutableData(bytes: &riBuf, length: riBuf.count*4), elementType: .float, shape: [1, 3, 3])

            for (classIdx, catFeats) in categoryFeatures.enumerated() {
                var feats = catFeats.features
                let fVal = try ORTValue(tensorData: NSMutableData(bytes: &feats, length: feats.count*4), elementType: .float,
                                         shape: [NSNumber(value: contextLength), 1, NSNumber(value: textFeatureDim)])
                var mask = catFeats.mask
                let mVal = try ORTValue(tensorData: NSMutableData(bytes: &mask, length: mask.count*8), elementType: .int64,
                                         shape: [1, NSNumber(value: contextLength)])
                let outputs = try session.run(
                    withInputs: ["image": preprocessed.imageArray, "depth_latents": dlVal, "ray_intrinsics": riVal,
                                 "text_features": fVal, "text_mask": mVal],
                    outputNames: ["pred_logits", "pred_boxes_2d", "pred_boxes_3d", "pred_conf_3d", "presence_logits"], runOptions: nil)
                func ef(_ v: ORTValue) throws -> [Float] { (try v.tensorData() as Data).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
                let logits = try ef(outputs["pred_logits"]!), boxes2d = try ef(outputs["pred_boxes_2d"]!)
                let boxes3d = try ef(outputs["pred_boxes_3d"]!), conf3d = try ef(outputs["pred_conf_3d"]!)
                let presence = try ef(outputs["presence_logits"]!)
                let S = nQueries, H = Float(inputSize), W = Float(inputSize), ps = sigmoid(presence[0])
                for q in 0..<S {
                    let s2d = sigmoid(logits[q]) * ps, s3d = sigmoid(conf3d[q])
                    if s2d < scoreThreshold { continue }
                    let b = q*4; let b3 = q*12
                    allProposals.append((s2d, s2d+0.5*s3d,
                        [boxes2d[b]*W, boxes2d[b+1]*H, boxes2d[b+2]*W, boxes2d[b+3]*H],
                        Array(boxes3d[b3..<b3+12]), classIdx))
                }
            }
        }

        if allProposals.isEmpty {
            print("[LocalInference] 0 detections in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent()-startTime))s")
            return DetectionResponse(boxes: [], mode: "local_onnx", predicted_intrinsics: nil)
        }
        let kept = batchedNMS(proposals: allProposals, iouThreshold: iouThreshold)
        let K = preprocessed.intrinsicsPadded
        var worldBoxes: [BBox3D] = []
        for idx in kept {
            let p = allProposals[idx]
            let d = decodeBox3D(box2d: p.box2d, box3dEncoded: p.box3d, K: K)
            let w = cameraToWorldTransform(center: [d.cx,d.cy,d.cz], dims: [d.w,d.l,d.h], quaternion: d.quaternion, cameraToWorld: capture.cameraToWorld)
            let ci = p.classId
            worldBoxes.append(BBox3D(label: ci < categories.count ? categories[ci] : "object",
                center: w.center, size: w.size, rotation: w.rotation,
                color: categoryColors[ci % categoryColors.count], score: p.score2d, n_frames: nil, projected_corners: nil))
        }
        if categories.count > 1 && worldBoxes.count > 1 { worldBoxes = crossClassNMS(worldBoxes, distance: nmsDistance) }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("[LocalInference] \(worldBoxes.count) detections in \(String(format: "%.1f", elapsed))s (\(categories.count) cats)")
        return DetectionResponse(boxes: worldBoxes, mode: "local_onnx", predicted_intrinsics: nil)
    }

    // MARK: - Preprocessing

    private struct PreprocessedInput {
        let imageArray: ORTValue; let depthArray: ORTValue; let intrinsicsArray: ORTValue
        let originalW: Int; let originalH: Int; let newW: Int; let newH: Int
        let padLeft: Int; let padTop: Int; let padRight: Int; let padBottom: Int
        let intrinsicsPadded: [[Float]]
    }

    private func preprocess(capture: CaptureData) throws -> PreprocessedInput {
        guard let uiImage = UIImage(data: capture.pngData), let cgImage = uiImage.cgImage else { throw LocalInferenceError.invalidImage }
        let origW = cgImage.width, origH = cgImage.height
        var K = capture.intrinsicK; K[1][2] = Float(origH) - K[1][2]
        let scale = Float(inputSize) / Float(max(origW, origH))
        let newW = Int(Float(origW) * scale), newH = Int(Float(origH) * scale)
        K[0][0] *= scale; K[0][2] *= scale; K[1][1] *= scale; K[1][2] *= scale
        let padLeft = (inputSize-newW)/2, padTop = (inputSize-newH)/2
        K[0][2] += Float(padLeft); K[1][2] += Float(padTop)
        let imageData = try createImageTensor(cgImage: cgImage, newW: newW, newH: newH, padLeft: padLeft, padTop: padTop)
        let depthData = try createDepthTensor(depthPng: capture.depthPngData, origW: origW, origH: origH, newW: newW, newH: newH, padLeft: padLeft, padTop: padTop)
        var kFlat: [Float] = []; for row in K { kFlat.append(contentsOf: row) }
        let kVal = try ORTValue(tensorData: NSMutableData(bytes: &kFlat, length: 36), elementType: .float, shape: [1, 3, 3])
        return PreprocessedInput(imageArray: imageData, depthArray: depthData, intrinsicsArray: kVal,
            originalW: origW, originalH: origH, newW: newW, newH: newH,
            padLeft: padLeft, padTop: padTop, padRight: inputSize-newW-padLeft, padBottom: inputSize-newH-padTop, intrinsicsPadded: K)
    }

    private func createImageTensor(cgImage: CGImage, newW: Int, newH: Int, padLeft: Int, padTop: Int) throws -> ORTValue {
        let ctx = CGContext(data: nil, width: newW, height: newH, bitsPerComponent: 8, bytesPerRow: newW*4,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let px = ctx.data else { throw LocalInferenceError.invalidImage }
        let ptr = px.bindMemory(to: UInt8.self, capacity: newW*newH*4)
        let n = 3*inputSize*inputSize; var buf = [Float](repeating: 0, count: n)
        for c in 0..<3 { let nz = -imgMean[c]/imgStd[c]; let o = c*inputSize*inputSize; for i in 0..<inputSize*inputSize { buf[o+i] = nz } }
        for y in 0..<newH { for x in 0..<newW {
            let si = (y*newW+x)*4; let dy = y+padTop; let dx = x+padLeft
            for c in 0..<3 { buf[c*inputSize*inputSize+dy*inputSize+dx] = (Float(ptr[si+c])/255.0-imgMean[c])/imgStd[c] }
        }}
        return try ORTValue(tensorData: NSMutableData(bytes: &buf, length: n*4), elementType: .float,
                           shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)])
    }

    private func createDepthTensor(depthPng: Data?, origW: Int, origH: Int, newW: Int, newH: Int, padLeft: Int, padTop: Int) throws -> ORTValue {
        let n = inputSize*inputSize; var buf = [Float](repeating: 0, count: n)
        if let dd = depthPng, let di = UIImage(data: dd)?.cgImage {
            let dW = di.width, dH = di.height
            let dc = CGContext(data: nil, width: dW, height: dH, bitsPerComponent: 16, bytesPerRow: dW*2,
                              space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            dc?.draw(di, in: CGRect(x: 0, y: 0, width: dW, height: dH))
            if let dp = dc?.data {
                let u16 = dp.bindMemory(to: UInt16.self, capacity: dW*dH)
                let sx = Float(dW)/Float(newW), sy = Float(dH)/Float(newH)
                for y in 0..<newH { for x in 0..<newW {
                    buf[(y+padTop)*inputSize+(x+padLeft)] = Float(u16[min(Int(Float(y)*sy),dH-1)*dW+min(Int(Float(x)*sx),dW-1)])/1000.0
                }}
            }
        }
        return try ORTValue(tensorData: NSMutableData(bytes: &buf, length: n*4), elementType: .float,
                           shape: [1, 1, NSNumber(value: inputSize), NSNumber(value: inputSize)])
    }

    // MARK: - 3D Decode

    private struct DecodedBox3D { let cx: Float; let cy: Float; let cz: Float; let w: Float; let l: Float; let h: Float; let quaternion: [Float] }

    private func decodeBox3D(box2d: [Float], box3dEncoded: [Float], K: [[Float]]) -> DecodedBox3D {
        let cx2d = (box2d[0]+box2d[2])/2, cy2d = (box2d[1]+box2d[3])/2
        let px = cx2d+box3dEncoded[0]*centerScale, py = cy2d+box3dEncoded[1]*centerScale
        let z = exp(box3dEncoded[2]/depthScale)
        let camX = (px-K[0][2])*z/K[0][0], camY = (py-K[1][2])*z/K[1][1]
        var dw = exp(box3dEncoded[3]/dimScale), dl = exp(box3dEncoded[4]/dimScale); let dh = exp(box3dEncoded[5]/dimScale)
        var rm = rotation6dToMatrix(Array(box3dEncoded[6..<12]))
        if dw > dl { let t=dw; dw=dl; dl=t
            let c0=[rm[0][0],rm[1][0],rm[2][0]], c2=[rm[0][2],rm[1][2],rm[2][2]]
            rm[0][0] = -c2[0]; rm[1][0] = -c2[1]; rm[2][0] = -c2[2]; rm[0][2]=c0[0]; rm[1][2]=c0[1]; rm[2][2]=c0[2]
        }
        if atan2(-rm[2][0],rm[0][0]) < 0 || atan2(-rm[2][0],rm[0][0]) > Float.pi-1e-4 {
            for r in 0..<3 { rm[r][0] = -rm[r][0]; rm[r][2] = -rm[r][2] }
        }
        return DecodedBox3D(cx: camX, cy: camY, cz: z, w: dw, l: dl, h: dh, quaternion: matrixToQuaternion(rm))
    }

    // MARK: - Camera-to-World

    private struct WorldBox { let center: [Float]; let size: [Float]; let rotation: [Float] }

    private func cameraToWorldTransform(center: [Float], dims: [Float], quaternion: [Float], cameraToWorld: [[Float]]) -> WorldBox {
        let T = matMul4x4(cameraToWorld, [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
        let cw = matVecMul4x4(T, [center[0],center[1],center[2],1])
        let fy3: [[Float]] = [[1,0,0],[0,-1,0],[0,0,1]]
        let rw = matMul3x3(matMul3x3(matMul3x3(extract3x3(from: cameraToWorld), fy3), quaternionToMatrix(quaternion)), fy3)
        let q = matrixToQuaternion(rw)
        return WorldBox(center: Array(cw[0..<3]), size: [dims[0],dims[2],dims[1]], rotation: [q[1],q[2],q[3],q[0]])
    }

    // MARK: - Math

    private func sigmoid(_ x: Float) -> Float { 1/(1+exp(-x)) }
    private func normalize(_ v: [Float]) -> [Float] { let l=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return l>1e-8 ? [v[0]/l,v[1]/l,v[2]/l]:[1,0,0] }
    private func cross(_ a: [Float], _ b: [Float]) -> [Float] { [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]] }
    private func dot3(_ a: [Float], _ b: [Float]) -> Float { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
    private func rotation6dToMatrix(_ d: [Float]) -> [[Float]] {
        let b1=normalize(Array(d[0..<3])); let dt=dot3(b1,Array(d[3..<6]))
        let b2=normalize([d[3]-dt*b1[0],d[4]-dt*b1[1],d[5]-dt*b1[2]]); return [b1,b2,cross(b1,b2)]
    }
    private func matrixToQuaternion(_ m: [[Float]]) -> [Float] {
        let q:[Float]=[sqrt(max(0,1+m[0][0]+m[1][1]+m[2][2]))/2,sqrt(max(0,1+m[0][0]-m[1][1]-m[2][2]))/2,
                        sqrt(max(0,1-m[0][0]+m[1][1]-m[2][2]))/2,sqrt(max(0,1-m[0][0]-m[1][1]+m[2][2]))/2]
        let i=q.enumerated().max(by:{$0.element<$1.element})!.offset
        var w:Float=0,x:Float=0,y:Float=0,z:Float=0
        switch i {
        case 0: w=q[0];x=(m[2][1]-m[1][2])/(4*w);y=(m[0][2]-m[2][0])/(4*w);z=(m[1][0]-m[0][1])/(4*w)
        case 1: x=q[1];w=(m[2][1]-m[1][2])/(4*x);y=(m[0][1]+m[1][0])/(4*x);z=(m[0][2]+m[2][0])/(4*x)
        case 2: y=q[2];w=(m[0][2]-m[2][0])/(4*y);x=(m[0][1]+m[1][0])/(4*y);z=(m[1][2]+m[2][1])/(4*y)
        default:z=q[3];w=(m[1][0]-m[0][1])/(4*z);x=(m[0][2]+m[2][0])/(4*z);y=(m[1][2]+m[2][1])/(4*z)
        }; return [w,x,y,z]
    }
    private func quaternionToMatrix(_ q: [Float]) -> [[Float]] {
        let w=q[0],x=q[1],y=q[2],z=q[3]
        return [[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
    }
    private func matMul3x3(_ a:[[Float]],_ b:[[Float]])->[[Float]]{var r=[[Float]](repeating:[Float](repeating:0,count:3),count:3);for i in 0..<3{for j in 0..<3{for k in 0..<3{r[i][j]+=a[i][k]*b[k][j]}}};return r}
    private func matMul4x4(_ a:[[Float]],_ b:[[Float]])->[[Float]]{var r=[[Float]](repeating:[Float](repeating:0,count:4),count:4);for i in 0..<4{for j in 0..<4{for k in 0..<4{r[i][j]+=a[i][k]*b[k][j]}}};return r}
    private func matVecMul4x4(_ m:[[Float]],_ v:[Float])->[Float]{var r=[Float](repeating:0,count:4);for i in 0..<4{for j in 0..<4{r[i]+=m[i][j]*v[j]}};return r}
    private func extract3x3(from m:[[Float]])->[[Float]]{[[m[0][0],m[0][1],m[0][2]],[m[1][0],m[1][1],m[1][2]],[m[2][0],m[2][1],m[2][2]]]}

    // MARK: - NMS

    private func batchedNMS(proposals:[(score2d:Float,scoreAll:Float,box2d:[Float],box3d:[Float],classId:Int)],iouThreshold:Float)->[Int]{
        var kept=[Int]()
        for cls in Set(proposals.map({$0.classId})){
            let idx=proposals.enumerated().filter({$0.element.classId==cls}).map({$0.offset}).sorted(by:{proposals[$0].scoreAll>proposals[$1].scoreAll})
            var sup=Set<Int>()
            for i in idx{if sup.contains(i){continue};kept.append(i);for j in idx where !sup.contains(j)&&j != i{if computeIoU(proposals[i].box2d,proposals[j].box2d)>iouThreshold{sup.insert(j)}}}
        }
        return kept.sorted{proposals[$0].scoreAll>proposals[$1].scoreAll}
    }

    private func computeIoU(_ a:[Float],_ b:[Float])->Float{
        let i=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        let u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i; return u>0 ? i/u:0
    }

    private func crossClassNMS(_ boxes:[BBox3D],distance:Float)->[BBox3D]{
        let s=boxes.sorted{$0.score>$1.score}; var k=[BBox3D]()
        for b in s{if !k.contains(where:{sqrt(pow($0.center[0]-b.center[0],2)+pow($0.center[1]-b.center[1],2)+pow($0.center[2]-b.center[2],2))<distance}){k.append(b)}}
        return k
    }
}

enum LocalInferenceError: LocalizedError {
    case modelNotFound(String); case notLoaded; case invalidImage
    var errorDescription: String? {
        switch self { case .modelNotFound(let n): return "Model not found: \(n)"; case .notLoaded: return "Models not loaded"; case .invalidImage: return "Failed to process image" }
    }
}
