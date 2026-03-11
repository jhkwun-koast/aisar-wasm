# ColormapRenderer WASM 모듈 구현 정리

**작성일**: 2026-03-11
**관련 PRD**: `prd_colormap-performance-optimization.md` (Tier 2), `prd_colormap-category-integration.md` (팔레트 시스템)

---

## 1. 개요

해류 컬러맵 렌더링의 성능 병목(Tier 2)을 해결하기 위해 WASM 기반 `ColormapRenderer` 모듈을 신규 개발했다.
기존 JS Worker의 렌더링 핫루프(좌표변환 + 공간인덱스 + 래스터라이징)를 Rust/WASM으로 교체하여
삼각형 래스터라이징을 직접 수행하고 RGBA 픽셀 버퍼를 반환한다.

추가로, PRD `prd_colormap-category-integration.md`의 W-1~W-5 요구사항에 따라
KHOA/Windy 듀얼 팔레트 시스템을 구현했다.

---

## 2. 파일 변경 내역

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `src/colormap.rs` | **신규** | ColormapRenderer 전체 구현 (약 750줄) |
| `src/lib.rs` | 수정 | `mod colormap;` 추가, 파티클 밀도 상수 조정, 테스트 수정 |

---

## 3. ColormapRenderer 구조

### 3.1 WASM 공개 API

```rust
#[wasm_bindgen]
impl ColormapRenderer {
    pub fn new() -> Self;
    pub fn load_mesh(&mut self, mesh_binary: &[u8]);
    pub fn load_uv(&mut self, uv_binary: &[u8], shuffled: bool);
    pub fn render(&self, extent: &[f64], width: u32, height: u32) -> Vec<u8>;
    pub fn set_palette(&mut self, name: &str);
    pub fn set_alpha(&mut self, alpha: u8);
    pub fn get_palette(&self) -> String;
    pub fn get_node_count(&self) -> usize;
    pub fn get_face_count(&self) -> usize;
}
```

### 3.2 내부 구조체

```rust
pub struct ColormapRenderer {
    mercator_x: Vec<f64>,       // WGS84→Mercator 사전 변환된 X좌표
    mercator_y: Vec<f64>,       // WGS84→Mercator 사전 변환된 Y좌표
    triangles: Vec<u32>,        // 삼각형 인덱스 (face_count × 3)
    speeds: Vec<f32>,           // 노드별 유속 (sqrt(u²+v²))
    node_count: usize,
    face_count: usize,
    lut_khoa: Vec<[u8; 4]>,     // KHOA 팔레트 LUT (1024 엔트리)
    lut_windy: Vec<[u8; 4]>,    // Windy 팔레트 LUT (1024 엔트리)
    active_palette: Palette,     // 현재 활성 팔레트
    alpha: u8,                  // 투명도 (기본 180)
    screen_x: Vec<f32>,         // render() 시 재사용되는 화면좌표 버퍼
    screen_y: Vec<f32>,
}
```

---

## 4. 렌더링 파이프라인

### 4.1 데이터 로드

1. **`load_mesh(mesh_binary)`**: SCHISM 바이너리 포맷 파싱
   - `[nodeCount:u32][triCount:u32][reserved:8bytes][nodes: N×16bytes][tris: T×12bytes]`
   - WGS84 좌표를 Web Mercator로 사전 변환하여 `mercator_x/y`에 저장
   - 화면좌표 버퍼(`screen_x/y`) 사전 할당

2. **`load_uv(uv_binary, shuffled)`**: UV 데이터 파싱
   - `[N × (u:f32, v:f32)]` little-endian
   - 각 노드의 유속 `speed = sqrt(u² + v²)` 계산하여 `speeds`에 저장

### 4.2 렌더링 (`render()`)

```
1. Affine 좌표 변환: mercatorXY → screenXY (단순 곱셈+덧셈)
2. 삼각형 순회 (face_count개):
   a. 화면좌표 3개 추출
   b. AABB 컬링: 뷰포트 밖이면 skip
   c. Bounding box 클리핑: 화면 경계 내로 제한
   d. 팔레트별 분기:
      - KHOA: 삼각형 평균 speed → LUT → flat color (삼각형 전체 동일 색상)
      - Windy: 픽셀별 barycentric 보간 speed → LUT → per-pixel color
   e. Scanline 래스터라이징: 바리센트릭 좌표로 내부 판정 + 픽셀 기록
3. RGBA 버퍼 반환 (width × height × 4 bytes)
```

### 4.3 핵심 최적화

| 최적화 | 설명 |
|--------|------|
| Mercator 사전 변환 | `load_mesh()` 시 1회만 계산, `render()` 시 affine만 수행 |
| LUT 사전 생성 | `new()` 시 KHOA + Windy 두 LUT 모두 생성 (1024 엔트리) |
| AABB 컬링 | 뷰포트 밖 삼각형 즉시 skip |
| 화면좌표 버퍼 재사용 | `screen_x/y`를 `render()` 호출 간 재활용 |
| 팔레트 전환 O(1) | `set_palette()`는 포인터만 전환, LUT 재생성 없음 |

---

## 5. 듀얼 팔레트 시스템

### 5.1 KHOA 팔레트 (해양조사원)

기존 `CurrentVectorWrapper`의 `calculate_color_from_speed()` 함수와 동일한 5단계 계단식 색상.
벡터 화살표 범례와 색상이 정확히 일치한다.

| 속도 구간 (m/s) | 색상 | RGB |
|-----------------|------|-----|
| 0.00 ~ 0.26 | 보라 | (159, 3, 202) |
| 0.26 ~ 0.51 | 파랑 | (3, 3, 202) |
| 0.51 ~ 0.76 | 초록 | (3, 202, 3) |
| 0.76 ~ 1.01 | 주황 | (254, 159, 3) |
| 1.01+ | 빨강 | (254, 3, 3) |

**렌더링 방식**: Per-triangle flat color
- 삼각형 3개 꼭짓점의 평균 speed 계산
- 해당 speed에 대응하는 단일 색상을 삼각형 전체에 적용
- 이유: 계단식 LUT + 픽셀 보간 시 삼각형 내부에 색 경계가 생겨 메시 격자와 불일치

### 5.2 Windy 팔레트

Windy.com 해류 시각화와 동일한 15단계 연속 보간 팔레트.

| 속도 (m/s) | 색상 | RGB |
|------------|------|-----|
| 0.00 ~ 0.20 | 어두운 남색 | (64, 77, 144) |
| 0.30 | 탁한 청록 | (61, 121, 109) |
| 0.40 | 어두운 초록 | (50, 140, 50) |
| 0.50 | 올리브/연두 | (141, 133, 49) |
| 0.60 | 갈색/주황 | (143, 115, 50) |
| 0.70 | 어두운 적갈색 | (116, 51, 68) |
| 0.80 | 탁한 보라 | (105, 68, 132) |
| 1.00 | 회색빛 파랑 | (66, 95, 133) |
| 1.20 | 회색빛 청록 | (74, 123, 132) |
| 1.40 | 밝은 회색 | (116, 135, 139) |
| 1.60 | 회색 | (144, 144, 144) |

**렌더링 방식**: Per-pixel barycentric interpolation
- 각 픽셀에서 바리센트릭 좌표로 speed를 보간
- 보간된 speed로 LUT 조회하여 색상 결정
- 연속 그라디언트이므로 삼각형 내 색 전환이 자연스러움

### 5.3 LUT 생성

- 크기: 1024 엔트리 (0.0 ~ 1.6 m/s 범위)
- KHOA: 계단식 — 속도 구간별 단일 색상 할당
- Windy: 연속식 — 15개 stop 간 선형 보간 (비균등 간격)
- 생성 시점: `ColormapRenderer::new()` 호출 시 양쪽 모두 생성
- `set_palette("khoa" | "windy")` 호출 시 활성 LUT 포인터만 전환

---

## 6. WASM-Client 연동 프로토콜

```
Main Thread → Colormap Worker:
  { type: 'loadMesh', meshBuffer: ArrayBuffer }    ← mesh 변경 시
  { type: 'loadUv', uvBuffer: ArrayBuffer }         ← 시간 스텝 변경 시
  { type: 'render', extent, width, height }         ← 뷰포트 변경 시 (32바이트만 전송)
  { type: 'updatePalette', palette: 'windy'|'khoa' } ← 팔레트 전환 시

Colormap Worker → WASM:
  wasmRenderer.load_mesh(meshBytes)
  wasmRenderer.load_uv(uvBytes, false)
  const pixels = wasmRenderer.render(extent, width, height)
  wasmRenderer.set_palette('khoa')

Worker → Main Thread:
  { type: 'rendered', bitmap: ImageBitmap }
```

---

## 7. 파티클 밀도 조정 (KEI 피드백)

파티클 가시성 개선을 위해 밀도 관련 상수를 조정했다.
JS 측에서 LINE_WIDTH를 3.0~3.5로 증가시키는 것과 함께 적용된다.

| 상수 | 변경 전 | 변경 후 | 효과 |
|------|---------|---------|------|
| `MAX_PARTICLES` | 7500 | 5000 | 전체 파티클 상한 33% 감소 |
| `MAX_PARTICLES_PER_CELL` | 8 | 5 | 셀 당 밀도 제한 37% 감소 |

---

## 8. 테스트

총 35개 테스트 통과:
- 기존 테스트 16개 (mesh 7개 + viewport/format 9개)
- 신규 colormap 테스트 19개

### 신규 테스트 목록

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_khoa_lut_step_colors` | KHOA LUT가 5단계 계단식 색상을 올바르게 생성 |
| `test_windy_lut_continuous` | Windy LUT가 연속 보간을 올바르게 생성 |
| `test_set_palette` | `set_palette()` 호출로 팔레트 전환 |
| `test_set_palette_invalid` | 잘못된 팔레트 이름 시 기본값 유지 |
| `test_set_alpha` | 알파값 설정 |
| `test_default_palette` | 기본 팔레트 = khoa |
| `test_new_renderer` | 초기 상태 검증 |
| `test_lut_boundary` | LUT 경계값 (0.0, max_speed) 검증 |
| `test_khoa_color_boundaries` | KHOA 색상 구간 경계 정확성 |
| `test_windy_interpolation` | Windy stop 사이 보간 정확성 |
| `test_render_empty` | 메시 없는 상태에서 render() 빈 버퍼 반환 |
| `test_load_mesh` | 메시 바이너리 파싱 + Mercator 변환 |
| `test_load_uv` | UV 바이너리 파싱 + speed 계산 |
| `test_render_basic` | 간단한 삼각형 래스터라이징 |
| `test_render_khoa_flat_color` | KHOA 팔레트에서 삼각형 내 동일 색상 |
| `test_render_windy_interpolated` | Windy 팔레트에서 픽셀 간 색상 차이 |
| `test_palette_switch_changes_output` | 팔레트 전환 시 렌더링 결과 변경 |
| `test_alpha_applied` | 알파값이 픽셀에 적용 |
| `test_render_viewport_culling` | 뷰포트 밖 삼각형 렌더링 제외 |

---

## 9. 기대 성능

| 단계 | 줌아웃 (49만 삼각형) | 줌인 (5000 삼각형) |
|------|---------------------|-------------------|
| 기존 JS | ~160ms | ~30ms |
| Tier 1 (JS 최적화) | ~60ms | ~15ms |
| **Tier 1+2 (WASM)** | **~10-15ms** | **~3ms** |

---

## 10. 향후 작업

| 항목 | 설명 |
|------|------|
| Client 연동 | JS Worker에서 WASM ColormapRenderer 호출 코드 구현 |
| 팔레트 전환 UI | 헤더 바 드롭다운에서 KHOA/Windy 전환 |
| 범례 연동 | 활성 팔레트에 따른 범례 표시/숨김 |
| 수온 컬러맵 | SCHISM 수온 데이터를 동일 파이프라인으로 렌더링 (향후) |
| SIMD 최적화 | wasm-simd 활용한 좌표 변환 4x 가속 (선택적) |
