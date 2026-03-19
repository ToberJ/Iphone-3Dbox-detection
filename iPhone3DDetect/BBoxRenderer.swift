import SceneKit
import simd

struct BoxMetadata {
    let label: String
    let score: Float
    let sizeMeters: simd_float3
}

class BBoxRenderer {
    private var boxNodes: [SCNNode] = []
    private var metadata: [SCNNode: BoxMetadata] = [:]
    private weak var scene: SCNScene?

    /// Tube radius for wireframe edges (meters). 4mm is visible on iPhone screens.
    var lineRadius: CGFloat = 0.004

    init(scene: SCNScene) {
        self.scene = scene
    }

    func addBoxes(_ boxes: [BBox3D]) {
        for box in boxes {
            let node = createBoxNode(box)
            scene?.rootNode.addChildNode(node)
            boxNodes.append(node)
            metadata[node] = BoxMetadata(
                label: box.label,
                score: box.score,
                sizeMeters: box.sizeSIMD
            )
        }
    }

    func clearBoxes() {
        for node in boxNodes {
            node.removeFromParentNode()
            metadata.removeValue(forKey: node)
        }
        boxNodes.removeAll()
    }

    var boxCount: Int { boxNodes.count }

    func updateLabels(cameraPosition: simd_float3, useMetric: Bool) {
        let factor: Float = useMetric ? 1.0 : 3.28084
        let unit = useMetric ? "m" : "ft"

        for node in boxNodes {
            guard let meta = metadata[node],
                  let labelNode = node.childNode(withName: "label", recursively: false),
                  let textGeom = labelNode.geometry as? SCNText
            else { continue }

            let dist = simd_distance(cameraPosition, node.simdWorldPosition) * factor
            let s = meta.sizeMeters * factor
            let pct = String(format: "%.0f%%", meta.score * 100)
            let distStr = dist < 10 ? String(format: "%.2f", dist) : String(format: "%.1f", dist)

            let line1 = "\(meta.label) (\(pct)) \(distStr)\(unit)"
            let line2 = String(format: "%.2f x %.2f x %.2f %@", s.x, s.y, s.z, unit)
            textGeom.string = "\(line1)\n\(line2)"

            let (bmin, bmax) = textGeom.boundingBox
            let scale: Float = 0.003
            let textWidth = Float(bmax.x - bmin.x) * scale
            labelNode.simdPosition = simd_float3(
                -textWidth / 2,
                meta.sizeMeters.y / 2 + 0.05,
                0
            )
        }
    }

    // MARK: - Node Creation

    private func createBoxNode(_ box: BBox3D) -> SCNNode {
        let parentNode = SCNNode()
        let center = box.centerSIMD
        parentNode.simdWorldPosition = simd_float3(center.x, center.y, -center.z)
        let q = box.quaternion
        let rhQuat = simd_quatf(ix: -q.imag.x, iy: -q.imag.y, iz: q.imag.z, r: q.real)
        parentNode.simdWorldOrientation = rhQuat * simd_quatf(angle: .pi / 2, axis: simd_float3(0, 1, 0))

        let c = box.displayColor
        let color = UIColor(red: CGFloat(c.r), green: CGFloat(c.g), blue: CGFloat(c.b), alpha: 1.0)
        let s = box.sizeSIMD

        addWireframeEdges(to: parentNode, size: s, color: color)
        addLabel(to: parentNode, box: box, boxSize: s, color: color)

        return parentNode
    }

    private func addWireframeEdges(to parent: SCNNode, size: simd_float3, color: UIColor) {
        let hx = size.x / 2, hy = size.y / 2, hz = size.z / 2

        let corners: [simd_float3] = [
            simd_float3(-hx, -hy, -hz), simd_float3( hx, -hy, -hz),
            simd_float3(-hx,  hy, -hz), simd_float3( hx,  hy, -hz),
            simd_float3(-hx, -hy,  hz), simd_float3( hx, -hy,  hz),
            simd_float3(-hx,  hy,  hz), simd_float3( hx,  hy,  hz),
        ]

        let edges: [(Int, Int)] = [
            (0,1),(2,3),(4,5),(6,7),
            (0,2),(1,3),(4,6),(5,7),
            (0,4),(1,5),(2,6),(3,7),
        ]

        for (a, b) in edges {
            let lineNode = createTube(from: corners[a], to: corners[b], color: color)
            parent.addChildNode(lineNode)
        }
    }

    private func createTube(from a: simd_float3, to b: simd_float3, color: UIColor) -> SCNNode {
        let diff = b - a
        let length = simd_length(diff)
        guard length > 0.0001 else { return SCNNode() }

        let cylinder = SCNCylinder(radius: lineRadius, height: CGFloat(length))
        cylinder.radialSegmentCount = 6
        let mat = SCNMaterial()
        mat.diffuse.contents = color
        mat.isDoubleSided = true
        cylinder.materials = [mat]

        let node = SCNNode(geometry: cylinder)
        let mid = (a + b) / 2
        node.simdPosition = mid

        let dir = simd_normalize(diff)
        let yAxis = simd_float3(0, 1, 0)
        let cross = simd_cross(yAxis, dir)
        let dot = simd_dot(yAxis, dir)
        if simd_length(cross) < 0.0001 {
            if dot < 0 {
                node.simdOrientation = simd_quatf(angle: .pi, axis: simd_float3(1, 0, 0))
            }
        } else {
            let angle = acos(max(-1, min(1, dot)))
            node.simdOrientation = simd_quatf(angle: angle, axis: simd_normalize(cross))
        }

        return node
    }

    private func addLabel(to parent: SCNNode, box: BBox3D, boxSize: simd_float3, color: UIColor) {
        let pct = String(format: "%.0f%%", box.score * 100)
        let placeholder = "\(box.label) (\(pct))"

        let textGeometry = SCNText(string: placeholder, extrusionDepth: 0.5)
        textGeometry.font = UIFont.systemFont(ofSize: 10, weight: .bold)
        textGeometry.firstMaterial?.diffuse.contents = color
        textGeometry.firstMaterial?.isDoubleSided = true
        textGeometry.flatness = 0.1

        let textNode = SCNNode(geometry: textGeometry)
        textNode.name = "label"
        let scale: Float = 0.003
        textNode.simdScale = simd_float3(scale, scale, scale)

        let (bmin, bmax) = textGeometry.boundingBox
        let textWidth = Float(bmax.x - bmin.x) * scale
        textNode.simdPosition = simd_float3(-textWidth / 2, boxSize.y / 2 + 0.05, 0)

        let billboardConstraint = SCNBillboardConstraint()
        billboardConstraint.freeAxes = .Y
        textNode.constraints = [billboardConstraint]

        parent.addChildNode(textNode)
    }
}

extension SCNVector3 {
    init(_ v: simd_float3) {
        self.init(x: v.x, y: v.y, z: v.z)
    }
}
