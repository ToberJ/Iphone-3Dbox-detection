import Foundation
import simd

struct BBox3D: Codable {
    let label: String
    let center: [Float]
    let size: [Float]
    let rotation: [Float]?
    let color: [Float]?
    let score: Float
    let n_frames: Int?

    var centerSIMD: simd_float3 {
        simd_float3(center[0], center[1], center[2])
    }

    var sizeSIMD: simd_float3 {
        simd_float3(size[0], size[1], size[2])
    }

    var quaternion: simd_quatf {
        guard let r = rotation, r.count >= 4 else { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
        return simd_quatf(ix: r[0], iy: r[1], iz: r[2], r: r[3])
    }

    var displayColor: (r: Float, g: Float, b: Float) {
        guard let c = color, c.count >= 3 else { return (0, 1, 0) }
        return (c[0], c[1], c[2])
    }
}

struct PredictedIntrinsics: Codable {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float

    /// Vertical FOV in degrees from fy and image height
    func fovY(imageHeight: Float) -> Float {
        return 2.0 * atan(imageHeight / (2.0 * fy)) * 180.0 / .pi
    }
}

struct DetectionResponse: Codable {
    let boxes: [BBox3D]
    let mode: String?
    let predicted_intrinsics: PredictedIntrinsics?
}

struct CaptureData {
    let pngData: Data
    let depthPngData: Data?
    let alignmentPostProcess: Bool
    let intrinsicK: [[Float]]
    let cameraToWorld: [[Float]]
}
