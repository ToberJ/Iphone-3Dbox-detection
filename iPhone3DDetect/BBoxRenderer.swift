import SceneKit
import simd

class BBoxRenderer {
    private var boxNodes: [SCNNode] = []
    private weak var scene: SCNScene?

    /// Tube radius for wireframe edges (meters). 4mm is visible on iPhone screens.
    var lineRadius: CGFloat = 0.004

    init(scene: SCNScene) {
        self.scene = scene
    }

    func showBoxes(_ boxes: [BBox3D]) {
        clearBoxes()
        for box in boxes {
            let node = createBoxNode(box)
            scene?.rootNode.addChildNode(node)
            boxNodes.append(node)
        }
    }

    func clearBoxes() {
        for node in boxNodes {
            node.removeFromParentNode()
        }
        boxNodes.removeAll()
    }

    private func createBoxNode(_ box: BBox3D) -> SCNNode {
        let parentNode = SCNNode()
        // LH→RH: negate Z of center
        let center = box.centerSIMD
        parentNode.simdWorldPosition = simd_float3(center.x, center.y, -center.z)
        // LH→RH quaternion (negate ix, iy) + 90° Y rotation (same API quirk as Quest)
        let q = box.quaternion
        let rhQuat = simd_quatf(ix: -q.imag.x, iy: -q.imag.y, iz: q.imag.z, r: q.real)
        parentNode.simdWorldOrientation = rhQuat * simd_quatf(angle: .pi / 2, axis: simd_float3(0, 1, 0))

        let c = box.displayColor
        let color = UIColor(red: CGFloat(c.r), green: CGFloat(c.g), blue: CGFloat(c.b), alpha: 1.0)
        let s = box.sizeSIMD

        addWireframeEdges(to: parentNode, size: s, color: color)
        addLabel(to: parentNode, text: "\(box.label) (\(String(format: "%.2f", box.score)))",
                 boxSize: s, color: color)

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

    private func addLabel(to parent: SCNNode, text: String, boxSize: simd_float3, color: UIColor) {
        let textGeometry = SCNText(string: text, extrusionDepth: 0.5)
        textGeometry.font = UIFont.systemFont(ofSize: 10, weight: .bold)
        textGeometry.firstMaterial?.diffuse.contents = color
        textGeometry.firstMaterial?.isDoubleSided = true
        textGeometry.flatness = 0.1

        let textNode = SCNNode(geometry: textGeometry)
        let scale: Float = 0.003
        textNode.simdScale = simd_float3(scale, scale, scale)

        let (min, max) = textGeometry.boundingBox
        let textWidth = Float(max.x - min.x) * scale
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
