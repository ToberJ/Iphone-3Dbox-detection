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
    private var colors: [SCNNode: UIColor] = [:]
    private(set) var selectedNode: SCNNode?
    private var selectionFill: SCNNode?
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
        deselectAll()
        for node in boxNodes {
            node.removeFromParentNode()
            metadata.removeValue(forKey: node)
            colors.removeValue(forKey: node)
        }
        boxNodes.removeAll()
    }

    var boxCount: Int { boxNodes.count }

    // MARK: - Selection

    /// Returns the top-level box parent if the tapped node belongs to a bbox, nil otherwise.
    func boxParent(for node: SCNNode) -> SCNNode? {
        var current: SCNNode? = node
        while let n = current {
            if boxNodes.contains(n) { return n }
            current = n.parent
        }
        return nil
    }

    func selectBox(_ node: SCNNode) {
        guard boxNodes.contains(node) else { return }
        deselectAll()
        selectedNode = node
        let color = colors[node] ?? .systemBlue
        let meta = metadata[node]
        let s = meta?.sizeMeters ?? simd_float3(0.1, 0.1, 0.1)

        let boxGeo = SCNBox(width: CGFloat(s.x), height: CGFloat(s.y), length: CGFloat(s.z), chamferRadius: 0)
        let mat = SCNMaterial()
        mat.diffuse.contents = color.withAlphaComponent(0.3)
        mat.isDoubleSided = true
        boxGeo.materials = [mat]

        let fill = SCNNode(geometry: boxGeo)
        fill.name = "selectionFill"
        node.addChildNode(fill)
        selectionFill = fill
    }

    /// Find the nearest box to a screen tap point by projecting box centers.
    func nearestBox(to screenPoint: CGPoint, in scnView: SCNView) -> SCNNode? {
        var bestNode: SCNNode?
        var bestDist: CGFloat = .greatestFiniteMagnitude

        for node in boxNodes {
            guard let meta = metadata[node] else { continue }
            let center = node.simdWorldPosition
            let projected = scnView.projectPoint(SCNVector3(center))
            guard projected.z > 0 && projected.z < 1 else { continue }

            let screenCenter = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))

            let offset = center + simd_float3(0, meta.sizeMeters.y / 2, 0)
            let projOffset = scnView.projectPoint(SCNVector3(offset))
            let halfScreenH = abs(CGFloat(projOffset.y - projected.y))
            let tapRadius = max(50, halfScreenH * 1.8)

            let dist = hypot(screenPoint.x - screenCenter.x, screenPoint.y - screenCenter.y)
            if dist < tapRadius && dist < bestDist {
                bestDist = dist
                bestNode = node
            }
        }
        return bestNode
    }

    func deselectAll() {
        selectionFill?.removeFromParentNode()
        selectionFill = nil
        selectedNode = nil
    }

    func removeBox(_ node: SCNNode) {
        if selectedNode == node { deselectAll() }
        node.removeFromParentNode()
        boxNodes.removeAll { $0 == node }
        metadata.removeValue(forKey: node)
        colors.removeValue(forKey: node)
    }

    func updateLabels(cameraPosition: simd_float3, useMetric: Bool) {
        let scale: Float = 0.003
        for node in boxNodes {
            guard let meta = metadata[node],
                  let labelNode = node.childNode(withName: "label", recursively: false),
                  let textGeom = labelNode.geometry as? SCNText
            else { continue }

            let pct = String(format: "%.0f%%", meta.score * 100)
            textGeom.string = "\(meta.label) (\(pct))"

            labelNode.simdScale = simd_float3(scale, scale, scale)
            let (bmin, bmax) = textGeom.boundingBox
            let textWidth = Float(bmax.x - bmin.x) * scale
            labelNode.simdPosition = simd_float3(
                -textWidth / 2,
                meta.sizeMeters.y / 2 + 0.05,
                0
            )
        }
    }

    /// Formatted caption string for a selected box (shown in screen-space overlay).
    func captionText(for node: SCNNode, cameraPosition: simd_float3, useMetric: Bool) -> String? {
        guard let meta = metadata[node] else { return nil }
        let factor: Float = useMetric ? 1.0 : 3.28084
        let unit = useMetric ? "m" : "ft"
        let dist = simd_distance(cameraPosition, node.simdWorldPosition) * factor
        let s = meta.sizeMeters * factor
        let pct = String(format: "%.0f%%", meta.score * 100)
        let distStr = dist < 10 ? String(format: "%.2f", dist) : String(format: "%.1f", dist)
        return "\(meta.label) (\(pct))  |  Distance: \(distStr)\(unit)  |  \(String(format: "%.2f × %.2f × %.2f", s.x, s.y, s.z)) \(unit)"
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
        colors[parentNode] = color

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
