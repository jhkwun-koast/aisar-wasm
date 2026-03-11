use wasm_bindgen::prelude::*;
use std::f64::consts::PI;

const MERC_SCALE: f64 = 20037508.34;
/// 색상 LUT 크기 (speed → RGBA 매핑)
const LUT_SIZE: usize = 1024;
/// LUT가 커버하는 최대 속도 (m/s)
const MAX_SPEED_LUT: f32 = 1.6;

// ── 팔레트 정의 ──────────────────────────────────────────────

#[inline]
fn lerp_channel(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t) as u8
}

#[inline]
fn lerp_rgb(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f64, alpha: u8) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        lerp_channel(c1.0, c2.0, t),
        lerp_channel(c1.1, c2.1, t),
        lerp_channel(c1.2, c2.2, t),
        alpha,
    ]
}

/// KHOA 팔레트: 5단계 계단식 (벡터 화살표 범례와 동일)
fn speed_to_rgba_khoa(speed: f32, alpha: u8) -> [u8; 4] {
    let s = speed as f64;
    let (r, g, b) = if s <= 0.26 {
        (127u8, 0u8, 127u8) // 보라색
    } else if s <= 0.51 {
        (3, 3, 202) // 파란색
    } else if s <= 0.76 {
        (0, 93, 0) // 초록색
    } else if s <= 1.01 {
        (233, 88, 0) // 주황색
    } else {
        (192, 0, 0) // 빨간색
    };
    [r, g, b, alpha]
}

/// Windy 해류 팔레트 (15 stops, 0.0–1.6 m/s)
const WINDY_STOPS: [(f32, (u8, u8, u8)); 15] = [
    (0.00, (64, 77, 144)),   // 어두운 남색
    (0.05, (64, 77, 144)),   // (동일 구간)
    (0.10, (64, 77, 144)),   // (동일 구간)
    (0.20, (64, 77, 144)),   // (동일 구간 — 0~0.2는 같은 색)
    (0.30, (61, 121, 109)),  // 탁한 청록
    (0.40, (50, 140, 50)),   // 어두운 초록
    (0.50, (141, 133, 49)),  // 올리브/연두
    (0.60, (143, 115, 50)),  // 갈색/주황
    (0.70, (116, 51, 68)),   // 어두운 적갈색
    (0.80, (105, 68, 132)),  // 탁한 보라
    (1.00, (66, 95, 133)),   // 회색빛 파랑
    (1.20, (74, 123, 132)),  // 회색빛 청록
    (1.40, (116, 135, 139)), // 밝은 회색
    (1.60, (144, 144, 144)), // 회색
    (1.60, (144, 144, 144)), // (동일)
];

/// Windy 팔레트: 10색 연속 보간 (piecewise linear)
fn speed_to_rgba_windy(speed: f32, alpha: u8) -> [u8; 4] {
    // 첫 구간 이하
    if speed <= WINDY_STOPS[0].0 {
        let c = WINDY_STOPS[0].1;
        return [c.0, c.1, c.2, alpha];
    }
    // 마지막 구간 이상
    let last = WINDY_STOPS.len() - 1;
    if speed >= WINDY_STOPS[last].0 {
        let c = WINDY_STOPS[last].1;
        return [c.0, c.1, c.2, alpha];
    }
    // 해당 구간 찾기
    for i in 0..last {
        let (s0, c0) = WINDY_STOPS[i];
        let (s1, c1) = WINDY_STOPS[i + 1];
        if speed <= s1 {
            let t = ((speed - s0) / (s1 - s0)) as f64;
            return lerp_rgb(c0, c1, t, alpha);
        }
    }
    // fallback (도달하지 않음)
    let c = WINDY_STOPS[last].1;
    [c.0, c.1, c.2, alpha]
}

// ── LUT 빌더 ─────────────────────────────────────────────────

fn build_khoa_lut(alpha: u8) -> Vec<[u8; 4]> {
    (0..LUT_SIZE)
        .map(|i| {
            let speed = (i as f32 / (LUT_SIZE - 1) as f32) * MAX_SPEED_LUT;
            speed_to_rgba_khoa(speed, alpha)
        })
        .collect()
}

fn build_windy_lut(alpha: u8) -> Vec<[u8; 4]> {
    (0..LUT_SIZE)
        .map(|i| {
            let speed = (i as f32 / (LUT_SIZE - 1) as f32) * MAX_SPEED_LUT;
            speed_to_rgba_windy(speed, alpha)
        })
        .collect()
}

// ── ColormapRenderer ──────────────────────────────────────────

/// 활성 팔레트
#[derive(Clone, Copy, Debug, PartialEq)]
enum Palette {
    Khoa,
    Windy,
}

#[wasm_bindgen]
pub struct ColormapRenderer {
    /// Mercator X 좌표 (EPSG:3857), 노드별
    mercator_x: Vec<f64>,
    /// Mercator Y 좌표 (EPSG:3857), 노드별
    mercator_y: Vec<f64>,
    /// 삼각형 인덱스 (flatten: [n0,n1,n2, n0,n1,n2, ...])
    triangles: Vec<u32>,
    /// 노드별 속도 √(u²+v²)
    speeds: Vec<f32>,
    node_count: usize,
    face_count: usize,
    /// 사전 생성된 LUT 2개
    lut_khoa: Vec<[u8; 4]>,
    lut_windy: Vec<[u8; 4]>,
    /// 현재 활성 팔레트
    active_palette: Palette,
    /// 오버레이 알파값
    alpha: u8,
    /// 재사용 가능한 스크린 좌표 버퍼
    screen_x: Vec<f32>,
    screen_y: Vec<f32>,
}

impl ColormapRenderer {
    /// 현재 활성 LUT 참조
    #[inline]
    fn active_lut(&self) -> &[[u8; 4]] {
        match self.active_palette {
            Palette::Khoa => &self.lut_khoa,
            Palette::Windy => &self.lut_windy,
        }
    }
}

#[wasm_bindgen]
impl ColormapRenderer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ColormapRenderer {
        let alpha = 200u8;
        ColormapRenderer {
            mercator_x: Vec::new(),
            mercator_y: Vec::new(),
            triangles: Vec::new(),
            speeds: Vec::new(),
            node_count: 0,
            face_count: 0,
            lut_khoa: build_khoa_lut(alpha),
            lut_windy: build_windy_lut(alpha),
            active_palette: Palette::Khoa, // 기본: KHOA (벡터 범례 일치)
            alpha,
            screen_x: Vec::new(),
            screen_y: Vec::new(),
        }
    }

    /// 팔레트 전환 — "windy" 또는 "khoa"
    /// 사전 생성된 LUT 포인터만 전환 (즉시 완료)
    pub fn set_palette(&mut self, name: &str) {
        match name {
            "windy" => self.active_palette = Palette::Windy,
            "khoa" => self.active_palette = Palette::Khoa,
            _ => {} // 알 수 없는 팔레트 → 무시
        }
    }

    /// 현재 팔레트 이름 반환
    pub fn get_palette(&self) -> String {
        match self.active_palette {
            Palette::Khoa => "khoa".to_string(),
            Palette::Windy => "windy".to_string(),
        }
    }

    /// 컬러맵 오버레이 알파값 설정 (0–255)
    /// 양쪽 LUT 모두 재생성
    pub fn set_alpha(&mut self, alpha: u8) {
        self.alpha = alpha;
        self.lut_khoa = build_khoa_lut(alpha);
        self.lut_windy = build_windy_lut(alpha);
    }

    /// mesh 바이너리 로드 + WGS84 → Mercator 사전 변환
    /// 포맷: [nodeCount:u32][triCount:u32][reserved:8][nodes: N×16 (lon:f64,lat:f64)][tris: T×12 (n0:u32,n1:u32,n2:u32)]
    pub fn load_mesh(&mut self, data: &[u8]) {
        if data.len() < 16 {
            return;
        }

        let node_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let tri_count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let nodes_start = 16usize;
        let nodes_end = nodes_start + node_count * 16;
        let tris_start = nodes_end;
        let tris_end = tris_start + tri_count * 12;

        if data.len() < tris_end {
            return;
        }

        // 노드 파싱 + WGS84 → Web Mercator
        let mut merc_x = Vec::with_capacity(node_count);
        let mut merc_y = Vec::with_capacity(node_count);

        for i in 0..node_count {
            let off = nodes_start + i * 16;
            let lon = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let lat = f64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());

            let x = lon * MERC_SCALE / 180.0;
            let y = ((PI / 4.0) + (lat.to_radians() / 2.0)).tan().ln() * MERC_SCALE / PI;

            merc_x.push(x);
            merc_y.push(y);
        }

        // 삼각형 인덱스 파싱
        let mut triangles = Vec::with_capacity(tri_count * 3);
        for i in 0..tri_count {
            let off = tris_start + i * 12;
            triangles.push(u32::from_le_bytes(data[off..off + 4].try_into().unwrap()));
            triangles.push(u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()));
            triangles.push(u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()));
        }

        // 인덱스 유효성 검증
        let max_idx = node_count as u32;
        for idx in &triangles {
            if *idx >= max_idx {
                return;
            }
        }

        self.mercator_x = merc_x;
        self.mercator_y = merc_y;
        self.triangles = triangles;
        self.node_count = node_count;
        self.face_count = tri_count;
        self.speeds = vec![0.0f32; node_count];
        self.screen_x = vec![0.0f32; node_count];
        self.screen_y = vec![0.0f32; node_count];
    }

    /// UV 바이너리 로드 → speed 계산
    /// shuffled=false: [u0:f32,v0:f32,u1:f32,v1:f32,...] (interleaved)
    /// shuffled=true:  [u0:f32,...,uN:f32, v0:f32,...,vN:f32] (separated)
    pub fn load_uv(&mut self, data: &[u8], shuffled: bool) {
        let count = self.node_count;
        if count == 0 {
            return;
        }

        let expected = count * 8;
        if data.len() < expected {
            return;
        }

        if shuffled {
            let u_section = count * 4;
            for i in 0..count {
                let u = f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
                let v = f32::from_le_bytes(
                    data[u_section + i * 4..u_section + i * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                self.speeds[i] = (u * u + v * v).sqrt();
            }
        } else {
            for i in 0..count {
                let off = i * 8;
                let u = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                let v = f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
                self.speeds[i] = (u * u + v * v).sqrt();
            }
        }
    }

    /// 뷰포트에 대한 컬러맵 RGBA 픽셀 버퍼 생성
    /// extent: [minX, minY, maxX, maxY] (Web Mercator, EPSG:3857)
    /// 반환: width × height × 4 bytes RGBA (ImageData 직접 사용 가능)
    pub fn render(&mut self, extent: &[f64], width: u32, height: u32) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut pixels = vec![0u8; w * h * 4];

        if self.node_count == 0 || self.face_count == 0 || extent.len() < 4 {
            return pixels;
        }

        let ext_min_x = extent[0];
        let ext_min_y = extent[1];
        let ext_max_x = extent[2];
        let ext_max_y = extent[3];

        let range_x = ext_max_x - ext_min_x;
        let range_y = ext_max_y - ext_min_y;
        if range_x <= 0.0 || range_y <= 0.0 {
            return pixels;
        }

        let scale_x = w as f64 / range_x;
        let scale_y = h as f64 / range_y;

        // ── Step 1: Affine 좌표 변환 (Mercator → Screen) ──
        for i in 0..self.node_count {
            self.screen_x[i] = ((self.mercator_x[i] - ext_min_x) * scale_x) as f32;
            self.screen_y[i] = ((ext_max_y - self.mercator_y[i]) * scale_y) as f32;
        }

        let w_f = w as f32;
        let h_f = h as f32;
        let lut = self.active_lut();
        let lut_scale = (LUT_SIZE - 1) as f32 / MAX_SPEED_LUT;

        // KHOA: 삼각형 단위 flat color (격자 일치)
        // Windy: 픽셀 단위 바리센트릭 보간 (부드러운 그라디언트)
        let use_flat = self.active_palette == Palette::Khoa;

        // ── Step 2: 삼각형 래스터라이징 ──
        for t in 0..self.face_count {
            let base = t * 3;
            let i0 = self.triangles[base] as usize;
            let i1 = self.triangles[base + 1] as usize;
            let i2 = self.triangles[base + 2] as usize;

            let ax = self.screen_x[i0];
            let ay = self.screen_y[i0];
            let bx = self.screen_x[i1];
            let by = self.screen_y[i1];
            let cx = self.screen_x[i2];
            let cy = self.screen_y[i2];

            // AABB 컬링
            let min_sx = ax.min(bx).min(cx);
            let max_sx = ax.max(bx).max(cx);
            let min_sy = ay.min(by).min(cy);
            let max_sy = ay.max(by).max(cy);

            if max_sx < 0.0 || min_sx >= w_f || max_sy < 0.0 || min_sy >= h_f {
                continue;
            }

            // 바리센트릭 분모 (퇴화 삼각형 스킵)
            let denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
            if denom.abs() < 1e-6 {
                continue;
            }
            let inv_denom = 1.0 / denom;

            let s0 = self.speeds[i0];
            let s1 = self.speeds[i1];
            let s2 = self.speeds[i2];

            // KHOA flat: 삼각형 3개 꼭짓점 평균 speed → 단일 색상
            let flat_color = if use_flat {
                let avg_speed = (s0 + s1 + s2) / 3.0;
                let lut_idx = ((avg_speed * lut_scale) as usize).min(LUT_SIZE - 1);
                Some(lut[lut_idx])
            } else {
                None
            };

            // 클리핑된 픽셀 범위
            let px_min = (min_sx.floor() as i32).max(0) as usize;
            let px_max = ((max_sx.ceil() as i32).min(w as i32 - 1)).max(0) as usize;
            let py_min = (min_sy.floor() as i32).max(0) as usize;
            let py_max = ((max_sy.ceil() as i32).min(h as i32 - 1)).max(0) as usize;

            // 바리센트릭 증분값 (scanline 최적화)
            let a01 = by - cy;
            let a12 = cy - ay;
            let b01 = cx - bx;
            let b12 = ax - cx;

            let dl1_dx = a01 * inv_denom;
            let dl2_dx = a12 * inv_denom;

            for py in py_min..=py_max {
                let py_f = py as f32 + 0.5;
                let px_start_f = px_min as f32 + 0.5;

                let mut l1 = (a01 * (px_start_f - cx) + b01 * (py_f - cy)) * inv_denom;
                let mut l2 = (a12 * (px_start_f - cx) + b12 * (py_f - cy)) * inv_denom;

                let row_offset = py * w;

                for px in px_min..=px_max {
                    let l3 = 1.0 - l1 - l2;

                    if l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0 {
                        let color = if let Some(c) = flat_color {
                            c
                        } else {
                            let speed = l1 * s0 + l2 * s1 + l3 * s2;
                            let lut_idx = ((speed * lut_scale) as usize).min(LUT_SIZE - 1);
                            lut[lut_idx]
                        };

                        let offset = (row_offset + px) * 4;
                        pixels[offset] = color[0];
                        pixels[offset + 1] = color[1];
                        pixels[offset + 2] = color[2];
                        pixels[offset + 3] = color[3];
                    }

                    l1 += dl1_dx;
                    l2 += dl2_dx;
                }
            }
        }

        pixels
    }

    /// 노드 수 반환 (디버그/검증용)
    pub fn get_node_count(&self) -> usize {
        self.node_count
    }

    /// 삼각형 수 반환
    pub fn get_face_count(&self) -> usize {
        self.face_count
    }
}

// ── 테스트 ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mesh_binary() -> Vec<u8> {
        let node_count: u32 = 3;
        let tri_count: u32 = 1;
        let mut buf = Vec::new();
        buf.extend_from_slice(&node_count.to_le_bytes());
        buf.extend_from_slice(&tri_count.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        for &(lon, lat) in &[(0.0f64, 0.0f64), (1.0, 0.0), (0.0, 1.0)] {
            buf.extend_from_slice(&lon.to_le_bytes());
            buf.extend_from_slice(&lat.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf
    }

    fn make_test_uv_binary(node_count: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        for i in 0..node_count {
            let u = (i as f32 + 1.0) * 0.1;
            let v = 0.0f32;
            buf.extend_from_slice(&u.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// LUT 인덱스 헬퍼
    fn speed_to_lut_idx(speed: f32) -> usize {
        ((speed / MAX_SPEED_LUT * (LUT_SIZE - 1) as f32) as usize).min(LUT_SIZE - 1)
    }

    #[test]
    fn test_new_default_palette_is_khoa() {
        let r = ColormapRenderer::new();
        assert_eq!(r.get_palette(), "khoa");
        assert_eq!(r.active_palette, Palette::Khoa);
    }

    #[test]
    fn test_set_palette_windy() {
        let mut r = ColormapRenderer::new();
        r.set_palette("windy");
        assert_eq!(r.get_palette(), "windy");
    }

    #[test]
    fn test_set_palette_unknown_ignored() {
        let mut r = ColormapRenderer::new();
        r.set_palette("unknown");
        assert_eq!(r.get_palette(), "khoa"); // 변경 없음
    }

    #[test]
    fn test_set_palette_instant_switch() {
        // set_palette는 LUT를 재생성하지 않고 포인터만 전환
        let mut r = ColormapRenderer::new();
        let khoa_first = r.active_lut()[0];
        r.set_palette("windy");
        let windy_first = r.active_lut()[0];
        // KHOA와 Windy의 speed=0 색상이 다름
        assert_ne!(khoa_first, windy_first);
        // 다시 KHOA로
        r.set_palette("khoa");
        assert_eq!(r.active_lut()[0], khoa_first);
    }

    #[test]
    fn test_khoa_palette_step_colors() {
        let lut = build_khoa_lut(255);

        // speed=0 → 보라색 (127, 0, 127)
        assert_eq!(lut[0][0..3], [127, 0, 127]);

        // speed=0.40과 0.50 모두 파란색 (계단식이므로 동일)
        let idx_040 = speed_to_lut_idx(0.40);
        let idx_050 = speed_to_lut_idx(0.50);
        assert_eq!(lut[idx_040], lut[idx_050]);
        assert_eq!(lut[idx_040][0..3], [3, 3, 202]);

        // 마지막 → 빨간색 (192, 0, 0)
        assert_eq!(lut[LUT_SIZE - 1][0..3], [192, 0, 0]);
    }

    #[test]
    fn test_khoa_step_boundary() {
        let lut = build_khoa_lut(255);
        // 0.25 → 보라색, 0.27 → 파란색 (경계 0.26)
        let idx_025 = speed_to_lut_idx(0.25);
        let idx_027 = speed_to_lut_idx(0.27);
        assert_eq!(lut[idx_025][0], 127); // 보라
        assert_eq!(lut[idx_027][0], 3); // 파란
    }

    #[test]
    fn test_windy_palette_endpoints() {
        let lut = build_windy_lut(255);

        // speed=0 → 어두운 남색 (64, 77, 144)
        assert_eq!(lut[0][0..3], [64, 77, 144]);

        // speed=1.6 (마지막 정지점) → 회색 (144, 144, 144)
        assert_eq!(lut[LUT_SIZE - 1][0..3], [144, 144, 144]);
    }

    #[test]
    fn test_windy_palette_continuous() {
        let lut = build_windy_lut(255);
        // speed=0.40과 0.50은 서로 다른 색 (연속 보간)
        let idx_040 = speed_to_lut_idx(0.40);
        let idx_050 = speed_to_lut_idx(0.50);
        assert_ne!(lut[idx_040], lut[idx_050]);
    }

    #[test]
    fn test_windy_palette_midpoint() {
        // speed=0.50 → 올리브/연두 (141, 133, 49) 정지점과 일치
        let rgba = speed_to_rgba_windy(0.50, 255);
        assert_eq!(rgba[0..3], [141, 133, 49]);
    }

    #[test]
    fn test_set_alpha_rebuilds_both_luts() {
        let mut r = ColormapRenderer::new();
        assert_eq!(r.lut_khoa[0][3], 200);
        assert_eq!(r.lut_windy[0][3], 200);

        r.set_alpha(128);
        assert_eq!(r.lut_khoa[0][3], 128);
        assert_eq!(r.lut_windy[0][3], 128);
        assert_eq!(r.lut_khoa[LUT_SIZE - 1][3], 128);
        assert_eq!(r.lut_windy[LUT_SIZE - 1][3], 128);
    }

    #[test]
    fn test_khoa_windy_same_speed_different_color() {
        // W-5: 두 팔레트가 같은 속도에서 다른 색을 내는지 확인
        let khoa = build_khoa_lut(255);
        let windy = build_windy_lut(255);

        let idx_030 = speed_to_lut_idx(0.30);
        // KHOA: 0.30 → 파란색(3,3,202), Windy: 0.30 → 탁한 청록(61,121,109)
        assert_eq!(khoa[idx_030][0..3], [3, 3, 202]);
        assert_ne!(khoa[idx_030][0..3], windy[idx_030][0..3]);
    }

    #[test]
    fn test_load_mesh() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&make_test_mesh_binary());
        assert_eq!(r.node_count, 3);
        assert_eq!(r.face_count, 1);
    }

    #[test]
    fn test_load_mesh_invalid() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&[0u8; 4]);
        assert_eq!(r.node_count, 0);
    }

    #[test]
    fn test_load_uv_interleaved() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&make_test_mesh_binary());
        r.load_uv(&make_test_uv_binary(3), false);
        assert!((r.speeds[0] - 0.1).abs() < 1e-5);
        assert!((r.speeds[1] - 0.2).abs() < 1e-5);
        assert!((r.speeds[2] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_load_uv_shuffled() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&make_test_mesh_binary());
        let mut buf = Vec::new();
        for i in 0..3 {
            buf.extend_from_slice(&((i as f32 + 1.0) * 0.1f32).to_le_bytes());
        }
        for _ in 0..3 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        r.load_uv(&buf, true);
        assert!((r.speeds[0] - 0.1).abs() < 1e-5);
        assert!((r.speeds[2] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_render_empty() {
        let mut r = ColormapRenderer::new();
        let pixels = r.render(&[0.0, 0.0, 1.0, 1.0], 10, 10);
        assert_eq!(pixels.len(), 400);
        assert!(pixels.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_render_produces_pixels() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&make_test_mesh_binary());
        r.load_uv(&make_test_uv_binary(3), false);
        let ext = [
            r.mercator_x.iter().cloned().fold(f64::MAX, f64::min) - 1000.0,
            r.mercator_y.iter().cloned().fold(f64::MAX, f64::min) - 1000.0,
            r.mercator_x.iter().cloned().fold(f64::MIN, f64::max) + 1000.0,
            r.mercator_y.iter().cloned().fold(f64::MIN, f64::max) + 1000.0,
        ];
        let pixels = r.render(&ext, 100, 100);
        assert!(pixels.chunks(4).any(|c| c[3] > 0));
    }

    #[test]
    fn test_render_outside_viewport() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&make_test_mesh_binary());
        r.load_uv(&make_test_uv_binary(3), false);
        let pixels = r.render(&[1e7, 1e7, 2e7, 2e7], 50, 50);
        assert!(pixels.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_render_palette_switch_changes_output() {
        let mut r = ColormapRenderer::new();
        r.load_mesh(&make_test_mesh_binary());
        r.load_uv(&make_test_uv_binary(3), false);
        let ext = [
            r.mercator_x.iter().cloned().fold(f64::MAX, f64::min) - 1000.0,
            r.mercator_y.iter().cloned().fold(f64::MAX, f64::min) - 1000.0,
            r.mercator_x.iter().cloned().fold(f64::MIN, f64::max) + 1000.0,
            r.mercator_y.iter().cloned().fold(f64::MIN, f64::max) + 1000.0,
        ];

        let pixels_khoa = r.render(&ext, 50, 50);
        r.set_palette("windy");
        let pixels_windy = r.render(&ext, 50, 50);

        // 동일 데이터, 다른 팔레트 → 다른 결과
        assert_ne!(pixels_khoa, pixels_windy);
    }
}
