import UIKit
import ARKit
import SceneKit
import ImageIO

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate, UITextFieldDelegate {

    private var sceneView: ARSCNView!
    private var bboxRenderer: BBoxRenderer!
    private var isDetecting = false

    // HUD
    private var statusLabel: UILabel!
    private var captureButton: UIButton!
    private var activityIndicator: UIActivityIndicatorView!
    private var settingsButton: UIButton!
    private var promptTextField: UITextField!
    private var promptBar: UIView!
    private var modeSegment: UISegmentedControl!
    private var captureSegment: UISegmentedControl!
    private var sendDepth = true
    private var alignmentPostProcess = true
    private var useMetric = true
    private var captureMode = 0  // 0=Single, 1=Multi-View, 2=Video
    private var clearButton: UIButton!
    private var videoFrames: [CaptureData] = []
    private var videoTimer: Timer?
    private let videoFrameCount = 15
    private let videoFrameInterval: TimeInterval = 0.2
    private var multiViewFrames: [CaptureData] = []
    private var multiViewDetectButton: UIButton!

    // 2D box drag
    private var boxOverlayView: UIView!
    private var dragStartPoint: CGPoint?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupSceneView()
        setupHUD()
        bboxRenderer = BBoxRenderer(scene: sceneView.scene)
        DetectionService.shared.warmUp()
    }

    private var isLiDARAvailable = false

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = []
        isLiDARAvailable = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        if isLiDARAvailable {
            config.frameSemantics = .sceneDepth
            print("[AR] LiDAR depth enabled")
        } else {
            print("[AR] No LiDAR - monocular fallback")
        }
        sceneView.session.run(config)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }

    // MARK: - Scene View

    private func setupSceneView() {
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.delegate = self
        sceneView.session.delegate = self
        sceneView.automaticallyUpdatesLighting = true
        view.addSubview(sceneView)

        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handleBoxDrag(_:)))
        panGesture.maximumNumberOfTouches = 1
        sceneView.addGestureRecognizer(panGesture)

        boxOverlayView = UIView()
        boxOverlayView.backgroundColor = UIColor.systemYellow.withAlphaComponent(0.15)
        boxOverlayView.layer.borderColor = UIColor.systemYellow.cgColor
        boxOverlayView.layer.borderWidth = 2.5
        boxOverlayView.isHidden = true
        boxOverlayView.isUserInteractionEnabled = false
        view.addSubview(boxOverlayView)
    }

    // MARK: - HUD

    private func setupHUD() {
        statusLabel = UILabel()
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.textColor = .white
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        statusLabel.font = .systemFont(ofSize: 14, weight: .medium)
        statusLabel.textAlignment = .center
        statusLabel.layer.cornerRadius = 8
        statusLabel.clipsToBounds = true
        statusLabel.text = " Ready - Tap Detect "
        view.addSubview(statusLabel)

        captureButton = UIButton(type: .system)
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.setTitle("  Detect  ", for: .normal)
        captureButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .bold)
        captureButton.setTitleColor(.white, for: .normal)
        captureButton.backgroundColor = UIColor.systemBlue
        captureButton.layer.cornerRadius = 25
        captureButton.addTarget(self, action: #selector(captureAndDetect), for: .touchUpInside)
        view.addSubview(captureButton)

        multiViewDetectButton = UIButton(type: .system)
        multiViewDetectButton.translatesAutoresizingMaskIntoConstraints = false
        multiViewDetectButton.setTitle("  Detect  ", for: .normal)
        multiViewDetectButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .bold)
        multiViewDetectButton.setTitleColor(.white, for: .normal)
        multiViewDetectButton.backgroundColor = UIColor.systemBlue
        multiViewDetectButton.layer.cornerRadius = 25
        multiViewDetectButton.addTarget(self, action: #selector(sendMultiViewDetection), for: .touchUpInside)
        multiViewDetectButton.isHidden = true
        view.addSubview(multiViewDetectButton)

        activityIndicator = UIActivityIndicatorView(style: .medium)
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.color = .white
        activityIndicator.hidesWhenStopped = true
        view.addSubview(activityIndicator)

        settingsButton = UIButton(type: .system)
        settingsButton.translatesAutoresizingMaskIntoConstraints = false
        settingsButton.setImage(UIImage(systemName: "gearshape.fill"), for: .normal)
        settingsButton.tintColor = .white
        settingsButton.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        settingsButton.layer.cornerRadius = 20
        settingsButton.addTarget(self, action: #selector(showSettings), for: .touchUpInside)
        view.addSubview(settingsButton)

        clearButton = UIButton(type: .system)
        clearButton.translatesAutoresizingMaskIntoConstraints = false
        clearButton.setImage(UIImage(systemName: "trash.fill"), for: .normal)
        clearButton.tintColor = .white
        clearButton.backgroundColor = UIColor.systemRed.withAlphaComponent(0.7)
        clearButton.layer.cornerRadius = 20
        clearButton.addTarget(self, action: #selector(clearAllBoxes), for: .touchUpInside)
        view.addSubview(clearButton)

        // Prompt bar for class input
        promptBar = UIView()
        promptBar.translatesAutoresizingMaskIntoConstraints = false
        promptBar.backgroundColor = UIColor.black.withAlphaComponent(0.75)
        promptBar.layer.cornerRadius = 12
        view.addSubview(promptBar)

        let promptLabel = UILabel()
        promptLabel.translatesAutoresizingMaskIntoConstraints = false
        promptLabel.text = "Classes:"
        promptLabel.textColor = .lightGray
        promptLabel.font = .systemFont(ofSize: 13, weight: .medium)
        promptBar.addSubview(promptLabel)

        promptTextField = UITextField()
        promptTextField.translatesAutoresizingMaskIntoConstraints = false
        promptTextField.text = DetectionService.shared.textPrompt
        promptTextField.textColor = .white
        promptTextField.font = .systemFont(ofSize: 14)
        promptTextField.returnKeyType = .done
        promptTextField.autocorrectionType = .no
        promptTextField.autocapitalizationType = .none
        promptTextField.attributedPlaceholder = NSAttributedString(
            string: "e.g. monitor.keyboard.chair",
            attributes: [.foregroundColor: UIColor.gray]
        )
        promptTextField.addTarget(self, action: #selector(promptChanged), for: .editingDidEnd)
        promptTextField.delegate = self
        promptBar.addSubview(promptTextField)

        modeSegment = UISegmentedControl(items: ["RGB", "RGBD", "RGBD+SAM2"])
        modeSegment.translatesAutoresizingMaskIntoConstraints = false
        modeSegment.selectedSegmentIndex = 2
        modeSegment.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        modeSegment.selectedSegmentTintColor = UIColor.systemBlue
        modeSegment.setTitleTextAttributes([.foregroundColor: UIColor.lightGray, .font: UIFont.systemFont(ofSize: 13, weight: .medium)], for: .normal)
        modeSegment.setTitleTextAttributes([.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: 13, weight: .bold)], for: .selected)
        modeSegment.addTarget(self, action: #selector(modeChanged), for: .valueChanged)
        view.addSubview(modeSegment)

        captureSegment = UISegmentedControl(items: ["Single", "Multi-View", "Video"])
        captureSegment.translatesAutoresizingMaskIntoConstraints = false
        captureSegment.selectedSegmentIndex = 0
        captureSegment.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        captureSegment.selectedSegmentTintColor = UIColor.systemOrange
        captureSegment.setTitleTextAttributes([.foregroundColor: UIColor.lightGray, .font: UIFont.systemFont(ofSize: 13, weight: .medium)], for: .normal)
        captureSegment.setTitleTextAttributes([.foregroundColor: UIColor.white, .font: UIFont.systemFont(ofSize: 13, weight: .bold)], for: .selected)
        captureSegment.addTarget(self, action: #selector(captureSegmentChanged), for: .valueChanged)
        view.addSubview(captureSegment)

        NSLayoutConstraint.activate([
            // Top row: prompt bar + mode segment + settings gear
            promptBar.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            promptBar.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            promptBar.trailingAnchor.constraint(equalTo: modeSegment.leadingAnchor, constant: -8),
            promptBar.heightAnchor.constraint(equalToConstant: 40),

            promptLabel.leadingAnchor.constraint(equalTo: promptBar.leadingAnchor, constant: 12),
            promptLabel.centerYAnchor.constraint(equalTo: promptBar.centerYAnchor),

            promptTextField.leadingAnchor.constraint(equalTo: promptLabel.trailingAnchor, constant: 8),
            promptTextField.trailingAnchor.constraint(equalTo: promptBar.trailingAnchor, constant: -12),
            promptTextField.centerYAnchor.constraint(equalTo: promptBar.centerYAnchor),

            modeSegment.centerYAnchor.constraint(equalTo: promptBar.centerYAnchor),
            modeSegment.trailingAnchor.constraint(equalTo: settingsButton.leadingAnchor, constant: -8),
            modeSegment.heightAnchor.constraint(equalToConstant: 32),

            settingsButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            settingsButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            settingsButton.widthAnchor.constraint(equalToConstant: 40),
            settingsButton.heightAnchor.constraint(equalToConstant: 40),

            clearButton.topAnchor.constraint(equalTo: settingsButton.bottomAnchor, constant: 8),
            clearButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            clearButton.widthAnchor.constraint(equalToConstant: 40),
            clearButton.heightAnchor.constraint(equalToConstant: 40),

            // Status below top bar
            statusLabel.topAnchor.constraint(equalTo: promptBar.bottomAnchor, constant: 8),
            statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 28),

            activityIndicator.centerYAnchor.constraint(equalTo: statusLabel.centerYAnchor),
            activityIndicator.trailingAnchor.constraint(equalTo: statusLabel.leadingAnchor, constant: -8),

            // Bottom row: capture segment (left) + detect button (right)
            captureSegment.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -18),
            captureSegment.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            captureSegment.heightAnchor.constraint(equalToConstant: 32),

            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            captureButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            captureButton.widthAnchor.constraint(equalToConstant: 120),
            captureButton.heightAnchor.constraint(equalToConstant: 50),

            multiViewDetectButton.bottomAnchor.constraint(equalTo: captureButton.bottomAnchor),
            multiViewDetectButton.trailingAnchor.constraint(equalTo: captureButton.leadingAnchor, constant: -12),
            multiViewDetectButton.widthAnchor.constraint(equalToConstant: 120),
            multiViewDetectButton.heightAnchor.constraint(equalToConstant: 50),
        ])
    }

    private func updateStatus(_ text: String, showSpinner: Bool = false) {
        DispatchQueue.main.async {
            self.statusLabel.text = "  \(text)  "
            if showSpinner {
                self.activityIndicator.startAnimating()
            } else {
                self.activityIndicator.stopAnimating()
            }
        }
    }

    // MARK: - Capture & Detect

    @objc private func captureAndDetect() {
        // Multi-view allows tapping again while "detecting" (collecting frames)
        if captureMode == 1 && !multiViewFrames.isEmpty {
            captureMultiViewFrame()
            return
        }

        guard !isDetecting else { return }
        guard sceneView.session.currentFrame != nil else {
            updateStatus("AR session not ready")
            return
        }

        isDetecting = true
        captureButton.isEnabled = false

        switch captureMode {
        case 1:
            startMultiViewCapture()
        case 2:
            startVideoCapture()
        default:
            singleCapture()
        }
    }

    private func singleCapture() {
        guard let frame = sceneView.session.currentFrame else { return }
        updateStatus("Capturing...", showSpinner: true)
        flashScreen()

        let captureData = extractCaptureData(from: frame)
        updateStatus("Detecting objects...", showSpinner: true)

        Task {
            do {
                let result = try await DetectionService.shared.detect(capture: captureData)
                await MainActor.run { handleResult(result) }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                }
            }
        }
    }

    // MARK: - Multi-View Capture

    private func startMultiViewCapture() {
        multiViewFrames = []
        guard let frame = sceneView.session.currentFrame else { return }
        flashScreen()
        let data = extractCaptureData(from: frame)
        multiViewFrames.append(data)
        print("[MultiView] Captured main frame")

        updateStatus("Main captured — move to a different angle and tap + Ref")
        captureButton.setTitle("  + Ref  ", for: .normal)
        captureButton.backgroundColor = UIColor.systemTeal
        captureButton.isEnabled = true
    }

    private func captureMultiViewFrame() {
        guard let frame = sceneView.session.currentFrame else { return }
        flashScreen()
        let data = extractCaptureData(from: frame)
        multiViewFrames.append(data)
        let refCount = multiViewFrames.count - 1
        print("[MultiView] Captured ref \(refCount)")

        multiViewDetectButton.isHidden = false
        updateStatus("Main + \(refCount) ref(s) — tap + Ref for more, or Detect to send")
    }

    @objc private func sendMultiViewDetection() {
        guard multiViewFrames.count >= 2 else { return }
        captureButton.isEnabled = false
        multiViewDetectButton.isHidden = true

        let frames = multiViewFrames
        multiViewFrames = []

        let mainFrame = frames[0]
        let refFrames = Array(frames[1...])

        updateStatus("Detecting (main + \(refFrames.count) ref)...", showSpinner: true)
        print("[MultiView] Sending main + \(refFrames.count) ref frame(s) to API")

        Task {
            do {
                let result = try await DetectionService.shared.detectWithReferences(main: mainFrame, references: refFrames)
                await MainActor.run {
                    handleResult(result)
                    resetCaptureButton()
                }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                    resetCaptureButton()
                }
            }
        }
    }

    private func resetCaptureButton() {
        let titles = ["  Detect  ", "  Main  ", "  Record  "]
        let colors = [UIColor.systemBlue, UIColor.systemGreen, UIColor.systemRed]
        captureButton.setTitle(titles[captureMode], for: .normal)
        captureButton.backgroundColor = colors[captureMode]
        multiViewDetectButton.isHidden = true
    }

    // MARK: - Video Capture

    private func startVideoCapture() {
        videoFrames = []
        updateStatus("Recording 0/\(videoFrameCount)...", showSpinner: true)
        flashScreen()

        captureVideoFrame()

        videoTimer = Timer.scheduledTimer(withTimeInterval: videoFrameInterval, repeats: true) { [weak self] timer in
            guard let self = self else { timer.invalidate(); return }
            self.captureVideoFrame()

            if self.videoFrames.count >= self.videoFrameCount {
                timer.invalidate()
                self.videoTimer = nil
                self.finishVideoCapture()
            }
        }
    }

    private func captureVideoFrame() {
        guard let frame = sceneView.session.currentFrame else { return }
        let data = extractCaptureData(from: frame)
        videoFrames.append(data)
        let count = videoFrames.count
        print("[Video] Captured frame \(count)/\(videoFrameCount)")
        updateStatus("Recording \(count)/\(videoFrameCount)...", showSpinner: true)
    }

    private func finishVideoCapture() {
        let frames = videoFrames
        videoFrames = []

        updateStatus("Detecting (\(frames.count) frames)...", showSpinner: true)
        print("[Video] Sending \(frames.count) frames to API")

        Task {
            do {
                let result = try await DetectionService.shared.detectMultiFrame(captures: frames)
                await MainActor.run { handleResult(result) }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                }
            }
        }
    }

    // MARK: - Result Handling

    private func handleResult(_ result: DetectionResponse) {
        let modeTag = "[\(result.mode ?? "?")]"
        if result.boxes.isEmpty {
            updateStatus("\(modeTag) No objects detected")
        } else {
            let summary = result.boxes.map { box -> String in
                let pct = String(format: "%.0f%%", box.score * 100)
                if let nf = box.n_frames {
                    return "\(box.label)(\(pct),\(nf)f)"
                }
                return "\(box.label)(\(pct))"
            }.joined(separator: ", ")
            updateStatus("\(modeTag) Found \(result.boxes.count): \(summary)")
            bboxRenderer.addBoxes(result.boxes)
        }
        finishDetection()
    }

    private func finishDetection() {
        isDetecting = false
        captureButton.isEnabled = true
        resetCaptureButton()
    }

    private func flashScreen() {
        let flash = UIView(frame: view.bounds)
        flash.backgroundColor = .white
        flash.alpha = 0.8
        view.addSubview(flash)
        UIView.animate(withDuration: 0.2) {
            flash.alpha = 0
        } completion: { _ in
            flash.removeFromSuperview()
        }
    }

    // MARK: - Real-time Label Updates

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard let pov = sceneView.pointOfView else { return }
        bboxRenderer.updateLabels(cameraPosition: pov.simdWorldPosition, useMetric: useMetric)
    }

    // MARK: - Clear Boxes

    @objc private func clearAllBoxes() {
        let count = bboxRenderer.boxCount
        bboxRenderer.clearBoxes()
        hideBoxOverlay()
        updateStatus("\(count) box(es) cleared")
    }

    // MARK: - 2D Box Drag-to-Detect

    @objc private func handleBoxDrag(_ gesture: UIPanGestureRecognizer) {
        let location = gesture.location(in: view)

        switch gesture.state {
        case .began:
            dragStartPoint = location
            boxOverlayView.frame = CGRect(origin: location, size: .zero)
            boxOverlayView.isHidden = false
            view.bringSubviewToFront(boxOverlayView)

        case .changed:
            guard let start = dragStartPoint else { return }
            let rect = CGRect(
                x: min(start.x, location.x),
                y: min(start.y, location.y),
                width: abs(location.x - start.x),
                height: abs(location.y - start.y)
            )
            boxOverlayView.frame = rect

        case .ended, .cancelled:
            guard let start = dragStartPoint else { return }
            dragStartPoint = nil
            let endPoint = location

            let minDrag: CGFloat = 30
            guard abs(endPoint.x - start.x) > minDrag && abs(endPoint.y - start.y) > minDrag else {
                boxOverlayView.isHidden = true
                return
            }

            captureWithBox2D(screenStart: start, screenEnd: endPoint)

        default:
            break
        }
    }

    private func captureWithBox2D(screenStart: CGPoint, screenEnd: CGPoint) {
        guard let frame = sceneView.session.currentFrame else { return }
        guard !isDetecting else { return }

        isDetecting = true
        flashScreen()

        let captureData = extractCaptureData(from: frame)

        let viewSize = sceneView.bounds.size
        let dt = frame.displayTransform(for: .landscapeRight, viewportSize: viewSize)
        let inv = dt.inverted()

        let normStart = CGPoint(x: screenStart.x / viewSize.width, y: screenStart.y / viewSize.height)
        let normEnd = CGPoint(x: screenEnd.x / viewSize.width, y: screenEnd.y / viewSize.height)
        let imgStart = normStart.applying(inv)
        let imgEnd = normEnd.applying(inv)

        let pixelBuffer = frame.capturedImage
        let origW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let origH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let imgW = origW * scale
        let imgH = origH * scale

        let x1 = Float(min(imgStart.x, imgEnd.x) * imgW)
        let y1 = Float(min(imgStart.y, imgEnd.y) * imgH)
        let x2 = Float(max(imgStart.x, imgEnd.x) * imgW)
        let y2 = Float(max(imgStart.y, imgEnd.y) * imgH)
        let box2D = [max(0, x1), max(0, y1), min(Float(imgW), x2), min(Float(imgH), y2)]

        print("[Box2D] Screen: (\(Int(screenStart.x)),\(Int(screenStart.y)))->(\(Int(screenEnd.x)),\(Int(screenEnd.y)))")
        print("[Box2D] Image pixels: [\(box2D[0]), \(box2D[1]), \(box2D[2]), \(box2D[3])] in \(Int(imgW))x\(Int(imgH))")

        updateStatus("Detecting from 2D box...", showSpinner: true)

        Task {
            do {
                let result = try await DetectionService.shared.detectWithBox2D(capture: captureData, box2D: box2D)
                await MainActor.run {
                    hideBoxOverlay()
                    handleResult(result)
                }
            } catch {
                await MainActor.run {
                    hideBoxOverlay()
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                }
            }
        }
    }

    private func hideBoxOverlay() {
        UIView.animate(withDuration: 0.3) {
            self.boxOverlayView.alpha = 0
        } completion: { _ in
            self.boxOverlayView.isHidden = true
            self.boxOverlayView.alpha = 1
        }
    }

    // MARK: - Extract ARFrame Data

    /// Max dimension for the image sent to the API. iPhone cameras are 4032x3024 which is too large.
    private let maxImageDimension: CGFloat = 1920

    private func extractCaptureData(from frame: ARFrame) -> CaptureData {
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
        let fullImage = UIImage(cgImage: cgImage)

        let origW = fullImage.size.width
        let origH = fullImage.size.height
        let scale = min(1.0, maxImageDimension / max(origW, origH))
        let newW = origW * scale
        let newH = origH * scale

        let resized: UIImage
        if scale < 1.0 {
            let renderer = UIGraphicsImageRenderer(size: CGSize(width: newW, height: newH))
            resized = renderer.image { _ in
                fullImage.draw(in: CGRect(x: 0, y: 0, width: newW, height: newH))
            }
        } else {
            resized = fullImage
        }
        let pngData = resized.pngData()!

        let intrinsics = frame.camera.intrinsics
        let fx = intrinsics.columns.0.x * Float(scale)
        let fy = intrinsics.columns.1.y * Float(scale)
        let cx = intrinsics.columns.2.x * Float(scale)
        let cyRaw = intrinsics.columns.2.y * Float(scale)
        // ARKit cy is from top edge (OpenCV). API expects from bottom edge (Unity).
        let cy = Float(newH) - cyRaw
        let K: [[Float]] = [
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ]

        // Convert ARKit right-handed cam2world to Unity left-handed: M_lh = S * M_rh * S
        // where S = diag(1,1,-1). Negate Z column + Z row of rotation + tz. det stays +1.
        let t = frame.camera.transform
        let camToWorld: [[Float]] = [
            [ t.columns.0.x,  t.columns.1.x, -t.columns.2.x,  t.columns.3.x],
            [ t.columns.0.y,  t.columns.1.y, -t.columns.2.y,  t.columns.3.y],
            [-t.columns.0.z, -t.columns.1.z,  t.columns.2.z, -t.columns.3.z],
            [0, 0, 0, 1],
        ]

        var depthPng: Data? = nil
        if sendDepth, let sceneDepth = frame.sceneDepth {
            depthPng = depthMapTo16BitPNG(sceneDepth.depthMap)
            print("[Capture] Depth: \(CVPixelBufferGetWidth(sceneDepth.depthMap))x\(CVPixelBufferGetHeight(sceneDepth.depthMap)), PNG: \(depthPng?.count ?? 0)B")
        } else {
            print("[Capture] Depth skipped (mode=\(sendDepth ? "depth" : "rgb-only"), hasLiDAR=\(frame.sceneDepth != nil))")
        }

        print("[Capture] Original: \(Int(origW))x\(Int(origH)), Resized: \(Int(newW))x\(Int(newH)), PNG: \(pngData.count/1024)KB")
        print("[Capture] K: fx=\(fx) fy=\(fy) cx=\(cx) cy=\(cy) (raw=\(cyRaw), H=\(Int(newH)))")

        return CaptureData(pngData: pngData, depthPngData: depthPng, alignmentPostProcess: alignmentPostProcess, intrinsicK: K, cameraToWorld: camToWorld)
    }

    private func depthMapTo16BitPNG(_ depthMap: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let floatPtr = base.assumingMemoryBound(to: Float32.self)
        let stride = rowBytes / 4

        var uint16Data = [UInt16](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let meters = floatPtr[y * stride + x]
                if meters > 0 && meters < 65.535 {
                    uint16Data[y * w + x] = UInt16(meters * 1000.0)
                }
            }
        }

        return uint16Data.withUnsafeMutableBytes { rawBuf -> Data? in
            guard let ptr = rawBuf.baseAddress else { return nil }
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageByteOrderInfo.order16Little.rawValue)
            guard let provider = CGDataProvider(data: NSData(bytes: ptr, length: w * h * 2)),
                  let cgImage = CGImage(width: w, height: h,
                                        bitsPerComponent: 16, bitsPerPixel: 16,
                                        bytesPerRow: w * 2,
                                        space: CGColorSpaceCreateDeviceGray(),
                                        bitmapInfo: bitmapInfo,
                                        provider: provider,
                                        decode: nil, shouldInterpolate: false,
                                        intent: .defaultIntent)
            else { return nil }

            let mutableData = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(dest, cgImage, nil)
            guard CGImageDestinationFinalize(dest) else { return nil }
            return mutableData as Data
        }
    }

    // MARK: - Capture Mode Toggle

    @objc private func captureSegmentChanged() {
        captureMode = captureSegment.selectedSegmentIndex
        multiViewFrames = []
        resetCaptureButton()
        let modeNames = ["Single", "Multi-View", "Video (\(videoFrameCount) frames)"]
        print("[CaptureMode] \(modeNames[captureMode])")
    }

    // MARK: - Depth Mode Toggle

    @objc private func modeChanged() {
        let idx = modeSegment.selectedSegmentIndex
        sendDepth = idx >= 1
        alignmentPostProcess = idx == 2
        let modes = ["RGB", "RGBD", "RGBD+SAM2"]
        print("[Mode] Switched to \(modes[idx])")
        if sendDepth && !isLiDARAvailable {
            updateStatus("No LiDAR on this device - will use monocular")
        }
    }

    // MARK: - Prompt Input

    @objc private func promptChanged() {
        if let text = promptTextField.text, !text.isEmpty {
            DetectionService.shared.textPrompt = text
            print("[Prompt] Updated to: \(text)")
        }
    }

    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        promptChanged()
        return true
    }

    // MARK: - Settings

    @objc private func showSettings() {
        let ds = DetectionService.shared
        let unitLabel = useMetric ? "Units: Meters → Feet" : "Units: Feet → Meters"
        let serverLabel = "Server: \(ds.serverName)"

        let alert = UIAlertController(title: "Settings", message: serverLabel, preferredStyle: .alert)

        alert.addTextField { tf in
            tf.text = String(ds.scoreThreshold)
            tf.placeholder = "Score threshold (0.0 - 1.0)"
            tf.keyboardType = .decimalPad
        }

        alert.addAction(UIAlertAction(title: "Save", style: .default) { _ in
            if let threshStr = alert.textFields?[0].text, let thresh = Float(threshStr) {
                ds.scoreThreshold = max(0, min(1, thresh))
            }
        })

        alert.addAction(UIAlertAction(title: "Switch to Modal", style: .default) { [weak self] _ in
            ds.apiUrl = DetectionService.modalUrl
            ds.warmUp()
            self?.updateStatus("Server: Modal")
            print("[Settings] Switched to Modal")
        })

        alert.addAction(UIAlertAction(title: "Switch to ngrok", style: .default) { [weak self] _ in
            ds.apiUrl = DetectionService.ngrokUrl
            self?.updateStatus("Server: ngrok")
            print("[Settings] Switched to ngrok")
        })

        alert.addAction(UIAlertAction(title: unitLabel, style: .default) { [weak self] _ in
            guard let self = self else { return }
            self.useMetric.toggle()
            let newUnit = self.useMetric ? "Meters" : "Feet"
            self.updateStatus("Units: \(newUnit)")
            print("[Settings] Units switched to \(newUnit)")
        })

        alert.addAction(UIAlertAction(title: "Clear Boxes", style: .destructive) { [weak self] _ in
            self?.bboxRenderer.clearBoxes()
            self?.updateStatus("Boxes cleared")
        })

        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))

        present(alert, animated: true)
    }
}
