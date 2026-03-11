# 해류 컬러맵 렌더링 성능 최적화 계획

**생성일**: 2026-03-10
**이슈**: #442
**상태**: 계획 수립

---

## 현재 문제

패닝/줌 시 컬러맵 레이어가 눈에 띄게 느리다.
Worker로 메인 스레드 블로킹은 해결했지만, 렌더링 자체가 느려서 뷰포트 변경 시 지연이 발생한다.

---

## 현재 병목 분석

뷰포트 변경(패닝/줌) 발생 시 매번 수행되는 작업:

| # | 작업 | 비용 | 빈도 | 비고 |
|---|------|------|------|------|
| 1 | `meshBuffer.slice(0)` + `uvBuffer.slice(0)` | **수 MB 복사** | 매 뷰포트 변경 | mesh는 변하지 않는데 매번 복사 |
| 2 | `parseMesh(meshBuffer)` | ~5ms | 매 뷰포트 변경 | mesh 파싱 결과 캐시 없음 |
| 3 | `lonToX/latToY` × nodeCount | **~20ms** (24만 노드) | 매 뷰포트 변경 | EPSG:4326→3857 매번 재계산 |
| 4 | 화면 좌표 변환 (affine) | ~5ms | 매 뷰포트 변경 | 3857→screen 변환 |
| 5 | 공간 인덱스 구축 | **~30ms** | 매 뷰포트 변경 | 셀 할당 매번 재구축 |
| 6 | 삼각형 렌더링 | **~50-100ms** | 매 뷰포트 변경 | Canvas Path 또는 ImageData |
| 7 | `new OffscreenCanvas()` | ~1ms | 매 뷰포트 변경 | 재사용 안 함 |

**총 렌더링 시간**: ~110-160ms per viewport change
**패닝 시 체감**: 연속적인 뷰포트 변경 → 렌더 요청 폭주 → Worker 큐 밀림 → 위치 동기화 지연

---

## 최적화 전략 (3단계)

### Tier 1: JS 최적화 (WASM 불필요, 즉시 적용 가능)

예상 효과: **렌더링 시간 ~60% 감소** (160ms → ~60ms)

| # | 최적화 | 절감 | 설명 |
|---|--------|------|------|
| A | **mesh/UV 캐시 분리** | ~5-10ms + 수 MB 복사 제거 | mesh는 Worker에 1회 전송 후 캐시. 뷰포트 변경 시 extent만 전송. |
| B | **Mercator 좌표 사전 계산** | **~20ms** | 4326→3857 변환은 mesh 로드 시 1회만. 뷰포트 변경 시 affine 변환만 수행. |
| C | **OffscreenCanvas 재사용** | ~1ms | Worker 내부에서 캔버스 재활용 |
| D | **디바운스 + 이전 결과 유지** | 체감 개선 | 패닝 중 이전 bitmap 표시, 150ms idle 후 재렌더링 |
| E | **speed 캐시** | ~5ms | UV 변경 시에만 speed 재계산, 뷰포트 변경 시 기존 speed 재사용 |

#### A. Worker 프로토콜 재설계

현재:
```
뷰포트 변경 → { type: 'render', meshBuffer(copy), uvBuffer(copy), extent }
                  ↑ 매번 수 MB 복사 + 파싱
```

개선:
```
mesh 변경 시 → { type: 'loadMesh', meshBuffer }     ← 1회 transfer
UV 변경 시  → { type: 'loadUv', uvBuffer }           ← 시간 스텝 변경 시만
뷰포트 변경 → { type: 'render', extent, width, height }  ← extent 32바이트만 전송
```

Worker 내부에서 mesh, mercator 좌표, speeds를 캐시.

#### B. Mercator 좌표 사전 계산

```javascript
// mesh 로드 시 1회만 계산
mercatorX = new Float64Array(nodeCount);  // lonToX(lon)
mercatorY = new Float64Array(nodeCount);  // latToY(lat)

// 뷰포트 변경 시 — 단순 affine (곱셈 + 덧셈만)
screenX[i] = (mercatorX[i] - extMinX) * scaleX;
screenY[i] = (extMaxY - mercatorY[i]) * scaleY;
```

lonToX는 단순 곱셈이지만, latToY는 `Math.log(Math.tan(...))` — 노드 24만개 × 삼각함수 2개 = **비용 대부분**

#### D. 디바운스 전략

```
패닝/줌 시작 → 이전 bitmap을 그대로 표시 (즉시 응답)
150ms idle   → Worker에 렌더 요청 전송
Worker 완료  → 새 bitmap으로 교체
```

사용자 체감: 패닝 중에는 이전 이미지가 고정되어 보이지만 UI가 멈추지 않음.
패닝 멈추면 150ms 후 새 컬러맵 표시.

---

### Tier 2: WASM 활용 (Tier 1 적용 후에도 느린 경우)

예상 효과: **렌더링 핫루프 ~5-10x 가속** (~60ms → ~10ms)

WASM으로 이동할 대상:
1. **affine 좌표 변환** — Float64Array × nodeCount의 단순 곱셈/덧셈 → SIMD 가능
2. **공간 인덱스 구축** — 셀 할당 루프 → typed array 기반 WASM
3. **ImageData 래스터라이징** — point-in-triangle + 색상 매핑 → 가장 큰 이득

#### WASM 인터페이스 설계 (안)

```rust
// lib.rs (또는 별도 colormap_wasm 모듈)

#[wasm_bindgen]
pub struct ColormapRenderer {
    mercator_x: Vec<f64>,
    mercator_y: Vec<f64>,
    triangles: Vec<u32>,
    speeds: Vec<f32>,
    node_count: usize,
    face_count: usize,
}

#[wasm_bindgen]
impl ColormapRenderer {
    /// mesh 로드 + Mercator 사전 변환
    pub fn load_mesh(&mut self, mesh_binary: &[u8]) { ... }

    /// UV 로드 → speed 계산
    pub fn load_uv(&mut self, uv_binary: &[u8], shuffled: bool) { ... }

    /// 뷰포트에 대한 컬러맵 픽셀 버퍼 생성
    /// 반환: RGBA ImageData (width × height × 4 bytes)
    pub fn render(&self, extent: &[f64], width: u32, height: u32) -> Vec<u8> { ... }
}
```

JS Worker 측:
```javascript
// Worker에서 WASM 렌더러 사용
const renderer = new ColormapRenderer();
renderer.load_mesh(meshBytes);       // mesh 변경 시
renderer.load_uv(uvBytes, false);    // UV 변경 시

// 뷰포트 변경 시 — WASM이 직접 ImageData 픽셀 생성
const pixels = renderer.render(extent, width, height);
const imgData = new ImageData(new Uint8ClampedArray(pixels), width, height);
ctx.putImageData(imgData, 0, 0);
```

#### WASM 렌더링 알고리즘

```
1. affine 좌표 변환: mercatorXY → screenXY (SIMD 가능)
2. 뷰포트 내 삼각형 필터링 (AABB 컬링)
3. 삼각형 래스터라이징:
   - 각 삼각형의 bounding box 내 픽셀 순회
   - 바리센트릭 좌표로 내부 판정 + speed 보간
   - LUT 조회로 RGBA 색상 매핑
   - ImageData 픽셀 버퍼에 직접 기록
```

핵심 장점: JS의 `ctx.fill()` / `putImageData` 오버헤드 없이 WASM이 직접 픽셀 버퍼를 생성.

---

### Tier 3: WebGL (최종 목표, 선택적)

예상 효과: **GPU 활용으로 ~1ms 렌더링**

삼각형 메시 렌더링은 GPU의 기본 작업이므로 WebGL이 궁극적 해법이다.
하지만 OL 레이어 통합, 셰이더 작성 등 구현 비용이 높아 장기 과제로 분류.

```
WebGL 접근:
1. 삼각형 인덱스 + 노드 좌표를 VBO/IBO로 GPU 업로드
2. 속도값을 attribute로 전달
3. Fragment shader에서 팔레트 색상 매핑
4. OL의 WebGL 레이어 또는 커스텀 WebGL 오버레이
```

---

## 구현 순서 제안

### Phase A: JS 최적화 (Tier 1) — 즉시 구현

| 순서 | 작업 | 파일 |
|------|------|------|
| A-1 | Worker 프로토콜 재설계: loadMesh/loadUv/render 분리 | `currentsColormapWorker.js`, `currentsColormapLayer.js` |
| A-2 | Mercator 좌표 사전 계산 (mesh 로드 시 1회) | `currentsColormapWorker.js` |
| A-3 | OffscreenCanvas + speeds 캐시 | `currentsColormapWorker.js` |
| A-4 | 디바운스 150ms + 이전 bitmap 유지 | `currentsColormapLayer.js` |
| A-5 | 검증: 패닝/줌 체감 개선 확인 |  |

### Phase B: WASM 래스터라이저 (Tier 2) — Tier 1 이후

| 순서 | 작업 | 파일 |
|------|------|------|
| B-1 | `ColormapRenderer` WASM 모듈 설계 + Rust 구현 | `lib.rs` (WASM 담당) |
| B-2 | Worker에서 WASM 렌더러 연동 | `currentsColormapWorker.js` |
| B-3 | JS 렌더링 → WASM 렌더링 전환 (폴백 유지) | `currentsColormapWorker.js` |
| B-4 | 성능 측정 + 비교 |  |

### ~~Phase C: WebGL (Tier 3)~~ — 폐기

대상 사용자 PC에 GPU 미탑재. WebGL 사용 불가.

---

## Tier 간 관계

```
Tier 1 (JS 최적화)
├── Worker 프로토콜 분리 (loadMesh/loadUv/render)    ← Tier 2에서도 그대로 유지
├── mesh/mercator 좌표 캐시                          ← Tier 2에서도 그대로 유지
├── 디바운스 + 이전 bitmap 유지                       ← Tier 2에서도 그대로 유지
└── JS 렌더링 핫루프 (좌표변환, 공간인덱스, 래스터)     ← Tier 2에서 WASM으로 교체
         │
         ▼
Tier 2 (WASM 래스터라이저)  — Tier 1 위에 누적, 롤백 아님
└── JS 렌더링 핫루프 → WASM ColormapRenderer.render() 교체
    (나머지 Tier 1 인프라는 100% 그대로)
```

---

## 기대 성능 비교

| 상태 | 줌아웃(49만 삼각형) | 줌인(5000 삼각형) | 패닝 체감 |
|------|---------------------|-------------------|-----------|
| **현재** (Worker, JS) | ~160ms | ~30ms | 랙 있음 |
| **Tier 1** (JS 최적화) | ~60ms | ~15ms | 디바운스로 체감 개선 |
| **Tier 1+2** (WASM 추가) | ~10-15ms | ~3ms | 거의 실시간 |

---

## WASM 담당자 협의 사항

Tier 2 진행 시 WASM 담당자와 협의할 내용:

1. **기존 `aisar_wasm` 크레이트에 추가** vs **별도 WASM 모듈로 분리**
   - 기존 크레이트: 빌드 파이프라인 재활용, 단 파일 크기 증가
   - 별도 모듈: 독립 빌드, 컬러맵 Worker에서만 로드
2. **`ColormapRenderer` 인터페이스**: load_mesh / load_uv / render 3메서드
3. **반환 포맷**: RGBA 픽셀 버퍼 (`Vec<u8>`) — JS에서 ImageData로 직접 사용
4. **SIMD 활용 여부**: wasm-simd 지원 시 좌표 변환 4x 가속 가능

---

## 결론

**Tier 1(JS 최적화)을 먼저 적용**하면 패닝/줌 체감이 상당히 개선될 것으로 예상된다.
특히 디바운스 + mesh/mercator 캐시만으로 뷰포트 변경 시 전송량이 수 MB → 32바이트로 줄고,
좌표 변환 비용이 ~20ms → ~5ms로 감소한다.

**Tier 1 → Tier 2는 롤백이 아니라 누적**이다.
Tier 1의 프로토콜 분리, 캐시, 디바운스는 Tier 2에서도 100% 그대로 유지되며,
Tier 2는 JS 렌더링 핫루프(좌표변환 + 공간인덱스 + 래스터라이징)만 WASM으로 교체한다.
따라서 Tier 1은 Tier 2의 전제 조건이기도 하다.
