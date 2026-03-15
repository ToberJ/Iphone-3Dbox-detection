import UIKit
import ARKit
import SceneKit

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {

    private var sceneView: ARSCNView!
    private var bboxRenderer: BBoxRenderer!
    private var isDetecting = false

    // HUD
    private var statusLabel: UILabel!
    private var captureButton: UIButton!
    private var activityIndicator: UIActivityIndicatorView!
    private var settingsButton: UIButton!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupSceneView()
        setupHUD()
        bboxRenderer = BBoxRenderer(scene: sceneView.scene)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = []
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
        statusLabel.text = " Ready - Tap Detect to capture "
        view.addSubview(statusLabel)

        captureButton = UIButton(type: .system)
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.setTitle("  Detect  ", for: .normal)
        captureButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .bold)
        captureButton.setTitleColor(.white, for: .normal)
        captureButton.backgroundColor = UIColor.systemBlue
        captureButton.layer.cornerRadius = 30
        captureButton.addTarget(self, action: #selector(captureAndDetect), for: .touchUpInside)
        view.addSubview(captureButton)

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

        NSLayoutConstraint.activate([
            statusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 32),

            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -30),
            captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captureButton.widthAnchor.constraint(equalToConstant: 140),
            captureButton.heightAnchor.constraint(equalToConstant: 60),

            activityIndicator.centerYAnchor.constraint(equalTo: statusLabel.centerYAnchor),
            activityIndicator.trailingAnchor.constraint(equalTo: statusLabel.leadingAnchor, constant: -8),

            settingsButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            settingsButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            settingsButton.widthAnchor.constraint(equalToConstant: 40),
            settingsButton.heightAnchor.constraint(equalToConstant: 40),
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
        guard !isDetecting else { return }
        guard let frame = sceneView.session.currentFrame else {
            updateStatus("AR session not ready")
            return
        }

        isDetecting = true
        captureButton.isEnabled = false
        updateStatus("Capturing...", showSpinner: true)

        flashScreen()

        let captureData = extractCaptureData(from: frame)

        updateStatus("Detecting objects...", showSpinner: true)

        Task {
            do {
                let boxes = try await DetectionService.shared.detect(capture: captureData)
                await MainActor.run {
                    if boxes.isEmpty {
                        updateStatus("No objects detected")
                    } else {
                        let summary = boxes.map { "\($0.label)(\(String(format: "%.0f%%", $0.score * 100)))" }.joined(separator: ", ")
                        updateStatus("Found \(boxes.count): \(summary)")
                        bboxRenderer.showBoxes(boxes)
                    }
                    finishDetection()
                }
            } catch {
                await MainActor.run {
                    updateStatus("Error: \(error.localizedDescription)")
                    finishDetection()
                }
            }
        }
    }

    private func finishDetection() {
        isDetecting = false
        captureButton.isEnabled = true
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

        print("[Capture] Original: \(Int(origW))x\(Int(origH)), Resized: \(Int(newW))x\(Int(newH)), PNG: \(pngData.count/1024)KB")
        print("[Capture] K: fx=\(fx) fy=\(fy) cx=\(cx) cy=\(cy) (raw=\(cyRaw), H=\(Int(newH)))")

        return CaptureData(pngData: pngData, intrinsicK: K, cameraToWorld: camToWorld)
    }

    // MARK: - Settings

    @objc private func showSettings() {
        let alert = UIAlertController(title: "Detection Settings", message: nil, preferredStyle: .alert)

        alert.addTextField { tf in
            tf.text = DetectionService.shared.apiUrl
            tf.placeholder = "API URL"
            tf.font = .systemFont(ofSize: 12)
        }
        alert.addTextField { tf in
            tf.text = DetectionService.shared.textPrompt
            tf.placeholder = "Text prompt (dot-separated)"
        }
        alert.addTextField { tf in
            tf.text = String(DetectionService.shared.scoreThreshold)
            tf.placeholder = "Score threshold (0.0 - 1.0)"
            tf.keyboardType = .decimalPad
        }

        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Save", style: .default) { _ in
            if let url = alert.textFields?[0].text, !url.isEmpty {
                DetectionService.shared.apiUrl = url
            }
            if let prompt = alert.textFields?[1].text, !prompt.isEmpty {
                DetectionService.shared.textPrompt = prompt
            }
            if let threshStr = alert.textFields?[2].text, let thresh = Float(threshStr) {
                DetectionService.shared.scoreThreshold = max(0, min(1, thresh))
            }
        })

        alert.addAction(UIAlertAction(title: "Clear Boxes", style: .destructive) { [weak self] _ in
            self?.bboxRenderer.clearBoxes()
            self?.updateStatus("Boxes cleared")
        })

        present(alert, animated: true)
    }
}
