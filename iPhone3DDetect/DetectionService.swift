import Foundation

class DetectionService {
    static let shared = DetectionService()

    static let modalUrl = "https://hwk18105962347--sam3-3d-api-sam3service-web.modal.run/detect3d_json"
    static let ngrokUrl = "https://diploidic-describably-anabelle.ngrok-free.dev/detect3d_json"

    var apiUrl = DetectionService.modalUrl
    var textPrompt = "monitor.keyboard.table.chair.computer"
    var scoreThreshold: Float = 0.5

    var isModal: Bool { apiUrl == DetectionService.modalUrl }
    var isNgrok: Bool { apiUrl == DetectionService.ngrokUrl }
    var serverName: String {
        if isModal { return "Modal" }
        if isNgrok { return "ngrok" }
        return "Custom"
    }

    private init() {}

    /// Pre-warm the Modal container so the first detection doesn't hit a ~90s cold start.
    func warmUp() {
        guard let baseUrl = URL(string: apiUrl)?.deletingLastPathComponent() else { return }
        let healthUrl = baseUrl.appendingPathComponent("health")
        print("[DetectionClient] Warming up: \(healthUrl)")
        URLSession.shared.dataTask(with: healthUrl) { _, response, error in
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            if let error = error {
                print("[DetectionClient] Warm-up failed: \(error.localizedDescription)")
            } else {
                print("[DetectionClient] Warm-up response: \(code)")
            }
        }.resume()
    }

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

    func detectWithReferences(main: CaptureData, references: [CaptureData]) async throws -> DetectionResponse {
        print("[DetectionClient] Multi-view ref: main + \(references.count) reference(s)")

        var body: [String: Any] = [
            "image_base64": main.pngData.base64EncodedString(),
            "intrinsic": ["K": main.intrinsicK],
            "camera_to_world": ["matrix_4x4": main.cameraToWorld],
            "text_prompt": textPrompt,
            "score_threshold": scoreThreshold,
            "alignment_post_process": main.alignmentPostProcess,
            "source": "iphone"
        ]

        if let depthData = main.depthPngData {
            body["depth_base64"] = depthData.base64EncodedString()
            print("[DetectionClient] Main: img=\(main.pngData.count/1024)KB, depth=\(depthData.count)B")
        } else {
            print("[DetectionClient] Main: img=\(main.pngData.count/1024)KB, no depth")
        }

        var refArray: [[String: Any]] = []
        for (i, ref) in references.enumerated() {
            var refDict: [String: Any] = [
                "intrinsic": ["K": ref.intrinsicK],
                "camera_to_world": ["matrix_4x4": ref.cameraToWorld]
            ]
            if let depthData = ref.depthPngData {
                refDict["depth_base64"] = depthData.base64EncodedString()
            }
            refDict["image_base64"] = ref.pngData.base64EncodedString()
            refArray.append(refDict)
            print("[DetectionClient] Ref \(i): img=\(ref.pngData.count/1024)KB, depth=\(ref.depthPngData?.count ?? 0)B")
        }
        body["reference_frames"] = refArray

        let jsonData = try JSONSerialization.data(withJSONObject: body)
        print("[DetectionClient] Multi-view ref payload: \(jsonData.count / 1024)KB to \(apiUrl)")

        var request = URLRequest(url: URL(string: apiUrl)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
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
        print("[DetectionClient] Multi-view ref detected \(result.boxes.count) objects, mode=\(result.mode ?? "unknown")")
        return result
    }
    func detectWithBox2D(capture: CaptureData, box2D: [Float]) async throws -> DetectionResponse {
        print("[DetectionClient] Box2D detect: box=\(box2D)")

        var body: [String: Any] = [
            "image_base64": capture.pngData.base64EncodedString(),
            "intrinsic": ["K": capture.intrinsicK],
            "camera_to_world": ["matrix_4x4": capture.cameraToWorld],
            "box_2d": box2D,
            "text_prompt": textPrompt,
            "score_threshold": scoreThreshold,
            "source": "iphone"
        ]

        if let depthData = capture.depthPngData {
            body["depth_base64"] = depthData.base64EncodedString()
            body["alignment_post_process"] = capture.alignmentPostProcess
        }

        let jsonData = try JSONSerialization.data(withJSONObject: body)
        print("[DetectionClient] Box2D payload: \(jsonData.count / 1024)KB to \(apiUrl)")

        var request = URLRequest(url: URL(string: apiUrl)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
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
        print("[DetectionClient] Box2D detected \(result.boxes.count) objects, mode=\(result.mode ?? "unknown")")
        return result
    }

    // MARK: - Upload Mode (no intrinsics, no pose, no depth)

    func detectUpload(pngData: Data) async throws -> DetectionResponse {
        print("[DetectionClient] Upload detect: \(pngData.count / 1024)KB")

        let body: [String: Any] = [
            "image_base64": pngData.base64EncodedString(),
            "text_prompt": textPrompt,
            "score_threshold": scoreThreshold,
            "mode": "upload"
        ]

        return try await sendRequest(body: body, label: "Upload")
    }

    func detectUploadWithBox2D(pngData: Data, box2D: [Float]) async throws -> DetectionResponse {
        print("[DetectionClient] Upload Box2D: box=\(box2D), \(pngData.count / 1024)KB")

        let body: [String: Any] = [
            "image_base64": pngData.base64EncodedString(),
            "box_2d": box2D,
            "score_threshold": scoreThreshold,
            "mode": "upload"
        ]

        return try await sendRequest(body: body, label: "Upload Box2D")
    }

    // MARK: - Shared Request Helper

    private func sendRequest(body: [String: Any], label: String) async throws -> DetectionResponse {
        let jsonData = try JSONSerialization.data(withJSONObject: body)
        print("[DetectionClient] \(label) payload: \(jsonData.count / 1024)KB to \(apiUrl)")

        var request = URLRequest(url: URL(string: apiUrl)!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
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
        print("[DetectionClient] \(label) detected \(result.boxes.count) objects, mode=\(result.mode ?? "unknown")")
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
