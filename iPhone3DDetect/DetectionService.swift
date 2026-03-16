import Foundation

class DetectionService {
    static let shared = DetectionService()

    var apiUrl = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"
    var textPrompt = "monitor.keyboard.table.chair.computer"
    var scoreThreshold: Float = 0.5

    private init() {}

    func detect(capture: CaptureData) async throws -> DetectionResponse {
        let base64 = capture.pngData.base64EncodedString()
        print("[DetectionClient] Image: \(capture.pngData.count / 1024)KB, base64: \(base64.count / 1024)KB")
        print("[DetectionClient] K: \(capture.intrinsicK)")
        print("[DetectionClient] cam2world: \(capture.cameraToWorld)")

        var body: [String: Any] = [
            "image_base64": base64,
            "intrinsic": ["K": capture.intrinsicK],
            "camera_to_world": ["matrix_4x4": capture.cameraToWorld],
            "text_prompt": textPrompt,
            "score_threshold": scoreThreshold,
            "source": "iphone"
        ]

        if let depthData = capture.depthPngData {
            let depthBase64 = depthData.base64EncodedString()
            body["depth_base64"] = depthBase64
            body["alignment_post_process"] = capture.alignmentPostProcess
            print("[DetectionClient] Depth: \(depthData.count)B, base64: \(depthBase64.count / 1024)KB, alignment_post_process=\(capture.alignmentPostProcess)")
        } else {
            print("[DetectionClient] No depth (monocular fallback)")
        }

        let jsonData = try JSONSerialization.data(withJSONObject: body)
        print("[DetectionClient] Sending \(jsonData.count / 1024)KB to \(apiUrl), prompt=\"\(textPrompt)\"")

        var request = URLRequest(url: URL(string: apiUrl)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        request.timeoutInterval = 60
        request.httpBody = jsonData

        let (data, response) = try await URLSession.shared.data(for: request)

        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
        print("[DetectionClient] Response status: \(statusCode), body size: \(data.count)")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let responseBody = String(data: data, encoding: .utf8) ?? "no body"
            print("[DetectionClient] ERROR: \(responseBody)")
            throw DetectionError.serverError(statusCode: statusCode, message: responseBody)
        }

        let responseText = String(data: data, encoding: .utf8) ?? ""
        print("[DetectionClient] Response: \(responseText.prefix(500))")

        let result = try JSONDecoder().decode(DetectionResponse.self, from: data)
        print("[DetectionClient] Detected \(result.boxes.count) objects, mode=\(result.mode ?? "unknown")")
        return result
    }

    func detectMultiFrame(captures: [CaptureData]) async throws -> DetectionResponse {
        print("[DetectionClient] Multi-frame: \(captures.count) frames")

        var framesArray: [[String: Any]] = []
        for (i, cap) in captures.enumerated() {
            var frameDict: [String: Any] = [
                "image_base64": cap.pngData.base64EncodedString(),
                "intrinsic": ["K": cap.intrinsicK],
                "camera_to_world": ["matrix_4x4": cap.cameraToWorld]
            ]
            if let depthData = cap.depthPngData {
                frameDict["depth_base64"] = depthData.base64EncodedString()
            }
            framesArray.append(frameDict)
            print("[DetectionClient] Frame \(i): img=\(cap.pngData.count/1024)KB, depth=\(cap.depthPngData?.count ?? 0)B")
        }

        let alignVal = captures.first?.alignmentPostProcess ?? false
        let body: [String: Any] = [
            "frames": framesArray,
            "text_prompt": textPrompt,
            "score_threshold": scoreThreshold,
            "alignment_post_process": alignVal,
            "source": "iphone"
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: body)
        print("[DetectionClient] Multi-frame payload: \(jsonData.count / 1024)KB to \(apiUrl)")

        var request = URLRequest(url: URL(string: apiUrl)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        request.timeoutInterval = 120
        request.httpBody = jsonData

        let (data, response) = try await URLSession.shared.data(for: request)

        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
        print("[DetectionClient] Response status: \(statusCode), body size: \(data.count)")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let responseBody = String(data: data, encoding: .utf8) ?? "no body"
            print("[DetectionClient] ERROR: \(responseBody)")
            throw DetectionError.serverError(statusCode: statusCode, message: responseBody)
        }

        let responseText = String(data: data, encoding: .utf8) ?? ""
        print("[DetectionClient] Response: \(responseText.prefix(500))")

        let result = try JSONDecoder().decode(DetectionResponse.self, from: data)
        print("[DetectionClient] Multi-frame detected \(result.boxes.count) objects, mode=\(result.mode ?? "unknown")")
        return result
    }
}

enum DetectionError: LocalizedError {
    case serverError(statusCode: Int, message: String)

    var errorDescription: String? {
        switch self {
        case .serverError(let code, let msg):
            return "Server error (\(code)): \(msg.prefix(200))"
        }
    }
}
