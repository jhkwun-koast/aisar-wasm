use rstar::{RTree, RTreeObject, PointDistance, AABB};

/// 메시 노드: Web Mercator 좌표 + 유속 데이터
#[derive(Debug, Clone)]
pub struct MeshNode {
    pub x: f64,
    pub y: f64,
    pub u: f64,
    pub v: f64,
}

/// 삼각형 참조: 노드 인덱스 3개 + AABB
#[derive(Debug, Clone)]
pub struct TriangleRef {
    pub indices: [u32; 3],
    pub aabb_min: [f64; 2],
    pub aabb_max: [f64; 2],
}

impl TriangleRef {
    pub fn new(indices: [u32; 3], nodes: &[MeshNode]) -> Self {
        let (i0, i1, i2) = (indices[0] as usize, indices[1] as usize, indices[2] as usize);
        let ax = nodes[i0].x; let ay = nodes[i0].y;
        let bx = nodes[i1].x; let by = nodes[i1].y;
        let cx = nodes[i2].x; let cy = nodes[i2].y;

        let min_x = ax.min(bx).min(cx);
        let min_y = ay.min(by).min(cy);
        let max_x = ax.max(bx).max(cx);
        let max_y = ay.max(by).max(cy);

        TriangleRef {
            indices,
            aabb_min: [min_x, min_y],
            aabb_max: [max_x, max_y],
        }
    }

    pub fn centroid(&self, nodes: &[MeshNode]) -> (f64, f64) {
        let (i0, i1, i2) = (self.indices[0] as usize, self.indices[1] as usize, self.indices[2] as usize);
        let cx = (nodes[i0].x + nodes[i1].x + nodes[i2].x) / 3.0;
        let cy = (nodes[i0].y + nodes[i1].y + nodes[i2].y) / 3.0;
        (cx, cy)
    }
}

impl RTreeObject for TriangleRef {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.aabb_min, self.aabb_max)
    }
}

impl PointDistance for TriangleRef {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        // AABB 기반 최소 거리 계산
        let dx = if point[0] < self.aabb_min[0] {
            self.aabb_min[0] - point[0]
        } else if point[0] > self.aabb_max[0] {
            point[0] - self.aabb_max[0]
        } else {
            0.0
        };
        let dy = if point[1] < self.aabb_min[1] {
            self.aabb_min[1] - point[1]
        } else if point[1] > self.aabb_max[1] {
            point[1] - self.aabb_max[1]
        } else {
            0.0
        };
        dx * dx + dy * dy
    }
}

const EPSILON: f64 = 1e-10;

/// 크로스 프로덕트 기반 point-in-triangle 판정
pub fn point_in_triangle(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
    let d1 = cross(px, py, ax, ay, bx, by);
    let d2 = cross(px, py, bx, by, cx, cy);
    let d3 = cross(px, py, cx, cy, ax, ay);

    let has_neg = (d1 < -EPSILON) || (d2 < -EPSILON) || (d3 < -EPSILON);
    let has_pos = (d1 > EPSILON) || (d2 > EPSILON) || (d3 > EPSILON);

    !(has_neg && has_pos)
}

#[inline]
fn cross(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

/// Barycentric 좌표 계산 후 (u, v) 보간
pub fn barycentric_interpolate(px: f64, py: f64, nodes: &[MeshNode], tri: &TriangleRef) -> Option<(f64, f64)> {
    let (i0, i1, i2) = (tri.indices[0] as usize, tri.indices[1] as usize, tri.indices[2] as usize);
    let (ax, ay) = (nodes[i0].x, nodes[i0].y);
    let (bx, by) = (nodes[i1].x, nodes[i1].y);
    let (cx, cy) = (nodes[i2].x, nodes[i2].y);

    let denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if denom.abs() < EPSILON {
        return None; // 퇴화 삼각형
    }

    let lambda1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom;
    let lambda2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom;
    let lambda3 = 1.0 - lambda1 - lambda2;

    let u = lambda1 * nodes[i0].u + lambda2 * nodes[i1].u + lambda3 * nodes[i2].u;
    let v = lambda1 * nodes[i0].v + lambda2 * nodes[i1].v + lambda3 * nodes[i2].v;

    Some((u, v))
}

/// R-tree에서 삼각형 탐색 → point_in_triangle → barycentric 좌표 반환
/// 반환값: (삼각형 인덱스는 미사용, u, v)
pub fn find_triangle<'a>(
    x: f64,
    y: f64,
    rtree: &'a RTree<TriangleRef>,
    nodes: &[MeshNode],
) -> Option<&'a TriangleRef> {
    let point = [x, y];
    let envelope = AABB::from_point(point);

    // AABB가 겹치는 삼각형 후보 탐색
    for tri in rtree.locate_in_envelope_intersecting(&envelope) {
        let (i0, i1, i2) = (tri.indices[0] as usize, tri.indices[1] as usize, tri.indices[2] as usize);
        if point_in_triangle(
            x, y,
            nodes[i0].x, nodes[i0].y,
            nodes[i1].x, nodes[i1].y,
            nodes[i2].x, nodes[i2].y,
        ) {
            return Some(tri);
        }
    }

    None
}

/// 삼각형 면적 계산 (외적의 절반)
pub fn triangle_area(tri: &TriangleRef, nodes: &[MeshNode]) -> f64 {
    let (i0, i1, i2) = (tri.indices[0] as usize, tri.indices[1] as usize, tri.indices[2] as usize);
    let (ax, ay) = (nodes[i0].x, nodes[i0].y);
    let (bx, by) = (nodes[i1].x, nodes[i1].y);
    let (cx, cy) = (nodes[i2].x, nodes[i2].y);
    0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)).abs()
}

/// 좌표 → (u, v) 반환, 삼각형 없으면 None
pub fn mesh_interpolate_uv(
    x: f64,
    y: f64,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
) -> Option<(f64, f64)> {
    let tri = find_triangle(x, y, rtree, nodes)?;
    barycentric_interpolate(x, y, nodes, tri)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_nodes() -> Vec<MeshNode> {
        vec![
            MeshNode { x: 0.0, y: 0.0, u: 1.0, v: 0.0 },  // 0
            MeshNode { x: 4.0, y: 0.0, u: 0.0, v: 1.0 },  // 1
            MeshNode { x: 2.0, y: 3.0, u: 0.0, v: 0.0 },  // 2
            MeshNode { x: 6.0, y: 3.0, u: 1.0, v: 1.0 },  // 3
            MeshNode { x: 8.0, y: 0.0, u: 0.5, v: 0.5 },  // 4
        ]
    }

    fn make_test_rtree(nodes: &[MeshNode]) -> RTree<TriangleRef> {
        let tris = vec![
            TriangleRef::new([0, 1, 2], nodes),  // 좌측 삼각형
            TriangleRef::new([1, 4, 3], nodes),  // 우측 삼각형
            TriangleRef::new([1, 3, 2], nodes),  // 중앙 삼각형
        ];
        RTree::bulk_load(tris)
    }

    #[test]
    fn test_point_in_triangle_inside() {
        // (0,0)-(4,0)-(2,3) 삼각형 내부 점
        assert!(point_in_triangle(2.0, 1.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
    }

    #[test]
    fn test_point_in_triangle_vertex() {
        // 꼭짓점 위
        assert!(point_in_triangle(0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
        assert!(point_in_triangle(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
        assert!(point_in_triangle(2.0, 3.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
    }

    #[test]
    fn test_point_in_triangle_edge() {
        // 변 위 (0,0)-(4,0)의 중점
        assert!(point_in_triangle(2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
    }

    #[test]
    fn test_point_in_triangle_outside() {
        // 삼각형 외부
        assert!(!point_in_triangle(5.0, 5.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
        assert!(!point_in_triangle(-1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 3.0));
    }

    #[test]
    fn test_barycentric_interpolate() {
        let nodes = make_test_nodes();
        let tri = TriangleRef::new([0, 1, 2], &nodes);

        // 꼭짓점 0에서는 node[0]의 값 (1.0, 0.0) 반환
        let result = barycentric_interpolate(0.0, 0.0, &nodes, &tri).unwrap();
        assert!((result.0 - 1.0).abs() < 1e-6);
        assert!((result.1 - 0.0).abs() < 1e-6);

        // 꼭짓점 1에서는 node[1]의 값 (0.0, 1.0) 반환
        let result = barycentric_interpolate(4.0, 0.0, &nodes, &tri).unwrap();
        assert!((result.0 - 0.0).abs() < 1e-6);
        assert!((result.1 - 1.0).abs() < 1e-6);

        // 꼭짓점 2에서는 node[2]의 값 (0.0, 0.0) 반환
        let result = barycentric_interpolate(2.0, 3.0, &nodes, &tri).unwrap();
        assert!((result.0 - 0.0).abs() < 1e-6);
        assert!((result.1 - 0.0).abs() < 1e-6);

        // 중심점 (2, 1) → 세 값의 평균에 가까워야 함
        let result = barycentric_interpolate(2.0, 1.0, &nodes, &tri).unwrap();
        assert!(result.0 >= 0.0 && result.0 <= 1.0);
        assert!(result.1 >= 0.0 && result.1 <= 1.0);
    }

    #[test]
    fn test_find_triangle() {
        let nodes = make_test_nodes();
        let rtree = make_test_rtree(&nodes);

        // 좌측 삼각형 내부
        let tri = find_triangle(1.5, 0.5, &rtree, &nodes);
        assert!(tri.is_some());

        // 우측 삼각형 내부
        let tri = find_triangle(7.0, 0.5, &rtree, &nodes);
        assert!(tri.is_some());

        // 전체 영역 외부
        let tri = find_triangle(-5.0, -5.0, &rtree, &nodes);
        assert!(tri.is_none());
    }

    #[test]
    fn test_mesh_interpolate_uv() {
        let nodes = make_test_nodes();
        let rtree = make_test_rtree(&nodes);

        // 삼각형 내부 → Some
        let result = mesh_interpolate_uv(1.5, 0.5, &rtree, &nodes);
        assert!(result.is_some());

        // 삼각형 외부 → None
        let result = mesh_interpolate_uv(-5.0, -5.0, &rtree, &nodes);
        assert!(result.is_none());
    }
}
