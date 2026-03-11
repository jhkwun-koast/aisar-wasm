# SCHISM 해류 파티클 엔진 리팩토링 (aisar-wasm)

**생성일**: 2026-02-24  
**상태**: 계획 수립 중  
**프로젝트**: aisar-wasm (Rust/WASM)  
**참조 문서**: `aisar-current-visualization-refactoring-review.md` 섹션 2, 4, 7

---

## 문제 요약

현재 Rust/WASM 파티클 엔진(`CurrentFlowWrapper`)에 구조적 문제 3가지가 있다:

1. **보간 부정확**: 포인트만 전달되고 삼각형 연결정보가 소실됨 → 외해 성긴 영역에서 IDW가 엉뚱한 노드 참조 → 구멍 발생
2. **고정 파티클 수**: 3,500개 고정 → 줌 아웃 시 듬성듬성, 줌 인 시 과잉
3. **메쉬 편향 리스폰**: 랜덤 리스폰이 해안가(삼각형 촘촘)에 편중, 외해는 희박

이 문서는 위 3가지를 해결하기 위한 WASM 쪽 변경 사항을 기술한다.

---

## 작업 범위

### 작업 1: 삼각형 메시 데이터 구조 도입

**현재**: `RTree<WeatherData>` — 포인트(lon, lat, u, v) 단위 R-tree  
**변경**: 삼각형 메시 구조 도입 — 노드 배열 + 삼각형 인덱스 배열 + 삼각형 바운딩박스 R-tree

#### 새 데이터 구조

```
MeshData:
  nodes: Vec<(f64, f64)>          // 노드 좌표 (lon, lat) — 약 30만개
  triangles: Vec<(u32, u32, u32)> // 삼각형 인덱스 (n0, n1, n2) — 약 58만개
  uv: Vec<(f64, f64)>             // 노드별 UV 값 — nodes와 동일 크기

TriangleRTree:
  삼각형 AABB(바운딩박스)로 R-tree 구축 — 초기화 시 1회
```

#### 새 공개 API

| 메서드 | 용도 | 호출 시점 |
|--------|------|----------|
| `load_mesh(nodes_buf, triangles_buf)` | 노드 좌표 + 삼각형 인덱스 로드, R-tree 구축 | 메시 변경 시 (수개월 1회) |
| `update_uv(uv_buf)` | UV 값만 갱신 (노드 순서 동일) | 시간 스텝 변경 시 |

- `nodes_buf`: `Float64Array` — `[lon0, lat0, lon1, lat1, ...]`
- `triangles_buf`: `Uint32Array` — `[n0, n1, n2, n0, n1, n2, ...]`
- `uv_buf`: `Float64Array` — `[u0, v0, u1, v1, ...]`

#### 제거 대상

- 기존 포인트 단위 R-tree (`RTree<WeatherData>`) → 삼각형 R-tree로 교체
- `TilingSystem` (주석 처리된 쿼드트리 LOD) → 삭제
- `mask_data` (육지 마스크) → 삼각형 밖 = 육지이므로 불필요
- `DBSCAN 클러스터링` (주석 처리) → 삭제

### 작업 2: Barycentric 보간으로 전환

**현재**: IDW, Bicubic, Nearest (Kriging/Bilinear 주석 처리)  
**변경**: Barycentric 보간 단일 방식

#### 보간 과정

```
파티클 위치(lon, lat)
  → 삼각형 R-tree에서 AABB 후보 탐색 (O(log N))
  → 후보 중 point_in_triangle 검사
  → 소속 삼각형의 3개 꼭짓점으로 Barycentric 가중 평균
  → (u, v) 반환
```

#### 선택 이유

- SCHISM이 P1 선형 요소 사용 → 모델의 물리적 가정과 정확히 일치
- IDW/Bicubic은 성긴 영역에서 엉뚱한 노드를 참조할 수 있음
- 계산량: 곱셈/나눗셈 몇 번 → 현재 Bicubic보다 빠름

#### 삼각형 못 찾는 경우 (= 육지)

파티클을 즉시 리스폰 처리. 기존 mask_data 로직 불필요.

### 작업 3: 화면 기준 파티클 밀도 동적 관리

**현재**: 고정 3,500개  
**변경**: 화면 크기 기반 동적 산출

#### 새 공개 API

| 메서드 | 용도 | 호출 시점 |
|--------|------|----------|
| `set_viewport(canvas_w, canvas_h, extent)` | 화면 크기·지리 범위 전달 | 줌/팬 변경 시 |

#### 파티클 수 산출

```
target_count = (canvas_w * canvas_h) / (target_spacing * target_spacing)
target_count = clamp(target_count, min_particles, max_particles)
```

#### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `target_spacing` | 25.0 | 화면 기준 파티클 간격 (px) |
| `min_particles` | 2,000 | 최소 파티클 수 |
| `max_particles` | 10,000 | 최대 파티클 수 |
| `transition_rate` | 100 | 프레임당 최대 파티클 증감 수 |

#### 점진적 전환

급격한 파티클 수 변동 시 깜빡임 방지를 위해 프레임당 `transition_rate`개씩 점진적으로 증감한다.

- 증가: 부족한 셀에 신규 파티클 생성
- 감소: 수명이 가장 짧은 파티클부터 강제 종료

### 작업 4: 화면 공간 균일 분포 리스폰

**현재**: 랜덤 좌표 리스폰  
**변경**: 화면 분할 셀 기반 균일 리스폰

#### 리스폰 셀 격자

```
화면을 (canvas_w / grid_cell_size) × (canvas_h / grid_cell_size) 셀로 분할
각 셀에 목표 파티클 수 = total_target / cell_count
```

#### 리스폰 로직

```
파티클 수명 종료 시:
  1. 현재 파티클이 속한 화면 셀 확인
  2. 전체 셀 중 가장 부족한 셀 탐색
  3. 해당 셀 내 랜덤 화면 좌표 → 지리 좌표 변환
  4. 삼각형 탐색:
     - 삼각형 있음 → 리스폰 확정
     - 삼각형 없음 → 같은 셀 내 재시도 (최대 3회)
     - 3회 실패 → 인접 셀로 이동
```

#### 뷰포트 정보 필요

리스폰 시 화면 좌표 → 지리 좌표 변환이 필요하므로, JS에서 `set_viewport()`로 extent를 전달받아야 한다. extent는 `[minLon, minLat, maxLon, maxLat]` 형식.

#### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `grid_cell_size` | 48.0 | 리스폰 균일화 셀 크기 (px) |
| `velocity_density_bias` | 0.1 | 유속 기반 밀도 가중 (0=균일, 0.3=강조) |

### 작업 5: 출력 데이터 경량화

**현재**: `getUpdateParticles()` → 전체 파티클을 Uint8Array로 직렬화 (좌표 + prev 10개 + 메타)  
**변경**: 파티클당 최소 데이터만 출력

#### 새 출력 포맷

반투명 덮기 트레일 방식(aisar 쪽)으로 전환하면 prev_coordinates 배열이 불필요해진다.

```
파티클당 출력:
  prev_x: f64     // 이전 프레임 화면 좌표 x
  prev_y: f64     // 이전 프레임 화면 좌표 y
  curr_x: f64     // 현재 프레임 화면 좌표 x
  curr_y: f64     // 현재 프레임 화면 좌표 y
  speed: f32      // 속도 (색상 매핑용)
  life_ratio: f32 // 남은 수명 비율 0~1 (페이드아웃용)
```

- 파티클당: 40 bytes (기존 대비 대폭 절감)
- 8,000개 기준: 320 KB/frame

#### 좌표 변환

WASM 내부에서 지리 좌표 → 화면 좌표 변환을 수행한다. `set_viewport()`에서 받은 extent와 canvas 크기로 선형 변환. JS에서 수신 후 바로 Canvas에 그리면 된다.

---

## 보존할 기존 구현

리팩토링 시 아래 튜닝된 값과 로직은 그대로 유지한다:

| 항목 | 설명 |
|------|------|
| 줌 레벨별 exaggeration 테이블 | 직접 튜닝한 속도 과장 값 |
| life ±30% 랜덤 | 일제 리스폰 깜빡임 방지 |
| 수명 끝 페이드아웃 | alpha + lineWidth 동시 감소 → life_ratio로 전달 |
| 속도 기반 색상 구간 | 0.26/0.51/0.76/1.01 구간 → speed로 전달 |

---

## 구현 순서

| 단계 | 작업 | 의존 |
|------|------|------|
| 1 | 삼각형 메시 구조 + R-tree 구축 (`load_mesh`) | 없음 |
| 2 | Barycentric 보간 전환 + 기존 IDW/Bicubic 제거 | 단계 1 |
| 3 | UV 분리 갱신 (`update_uv`) | 단계 1 |
| 4 | `set_viewport()` API + 화면 좌표 변환 | 없음 |
| 5 | 동적 파티클 수 관리 | 단계 4 |
| 6 | 균일 분포 리스폰 | 단계 4, 5 |
| 7 | 출력 포맷 경량화 (prev_coordinates 제거) | 단계 4 |
| 8 | 미사용 코드 정리 (mask_data, TilingSystem, DBSCAN) | 전체 완료 후 |

---

## aisar (JS) 쪽 연동 사항

이 문서의 변경사항은 aisar 쪽에서 다음 작업이 선행/병행되어야 한다:

- **서버**: 바이너리 API 제공 (`/schism/mesh`, `/schism/uv`) → aisar 쪽 문서 참조
- **JS**: `load_mesh()`, `update_uv()`, `set_viewport()` 호출부 변경
- **JS**: 새 출력 포맷에 맞춰 Canvas 렌더링 로직 변경 (반투명 덮기 트레일)
