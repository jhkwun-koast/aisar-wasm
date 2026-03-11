# 파티클 해안선 사각형 클러스터링 버그 수정 — 면적 가중 삼각형 선택

**생성일**: 2026-02-24
**상태**: 구현 대기
**관련 파일**: `lib.rs`, `mesh.rs`

---

## 문제 요약

이전 Fix(폴백 랜덤화, Reservoir Sampling, 삼각형 직접 선택) 적용 후에도 해안선을 따라 **사각형 형태의 파티클 클러스터**가 발생한다. 좌측 상단 해안가에 분홍색 사각형 밀집, 중앙 만(灣) 내부 파티클 희박, 외해 영역 파티클 과소.

---

## 근본 원인

`spawn_in_random_triangle_viewport`와 `spawn_in_random_triangle_global`이 **삼각형 개수 기준 균등 확률**로 선택한다.

SCHISM 비정형 메시는 해안가에 소형 삼각형이 수천 개, 외해에 대형 삼각형이 수 개뿐이다. 개수 기준 균등 선택 시 해안가 삼각형 선택 확률이 99%+로, 폴백 경로를 탈 때마다 해안가 특정 셀에 파티클이 집중되어 사각형 클러스터가 형성된다.

---

## 수정 지시

### 1. `mesh.rs` — `triangle_area` 헬퍼 함수 추가

`mesh_interpolate_uv` 함수 바로 위에 삼각형 면적 계산 함수를 추가한다.

- 외적(cross product)의 절반으로 면적 계산: `0.5 * |((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))|`
- `pub fn triangle_area(tri: &TriangleRef, nodes: &[MeshNode]) -> f64`
- 퇴화 삼각형(면적 ≈ 0)은 호출 측에서 필터링

### 2. `lib.rs` — import 수정 (line 24)

`use crate::mesh::{...}` 에 `triangle_area`를 추가한다.

### 3. `lib.rs` — `spawn_in_random_triangle_viewport` 수정 (line 2252 부근)

현재 개수 기준 Reservoir Sampling을 **면적 가중 Reservoir Sampling**으로 교체한다.

**현재 코드 (교체 대상)**:
```rust
// Reservoir Sampling으로 랜덤 삼각형 선택
let mut selected: Option<&TriangleRef> = None;
let mut count: usize = 0;
for tri in rtree.locate_in_envelope_intersecting(&envelope) {
    count += 1;
    if generate_random_usize(0, count) == 0 {
        selected = Some(tri);
    }
}
```

**교체할 코드**:
```rust
// 면적 가중 Reservoir Sampling
let mut selected: Option<&TriangleRef> = None;
let mut cumulative_area: f64 = 0.0;
for tri in rtree.locate_in_envelope_intersecting(&envelope) {
    let area = triangle_area(tri, nodes);
    if area < 1e-20 { continue; } // 퇴화 삼각형 무시
    cumulative_area += area;
    // area / cumulative_area 확률로 현재 삼각형 선택
    if generate_random_f64(0.0, cumulative_area) < area {
        selected = Some(tri);
    }
}
```

함수 doc comment도 "면적 가중"임을 명시하도록 갱신한다.

### 4. `lib.rs` — `spawn_in_random_triangle_global` 수정 (line 2303 부근)

동일하게 면적 가중 Reservoir Sampling으로 교체한다.

**현재 코드 (교체 대상)**:
```rust
let total = rtree.size();
if total == 0 {
    return None;
}
let target_idx = generate_random_usize(0, total);
let tri = rtree.iter().nth(target_idx)?;
```

**교체할 코드**:
```rust
if rtree.size() == 0 {
    return None;
}
// 면적 가중 Reservoir Sampling
let mut selected: Option<&TriangleRef> = None;
let mut cumulative_area: f64 = 0.0;
for tri in rtree.iter() {
    let area = triangle_area(tri, nodes);
    if area < 1e-20 { continue; }
    cumulative_area += area;
    if generate_random_f64(0.0, cumulative_area) < area {
        selected = Some(tri);
    }
}
let tri = selected?;
```

함수 doc comment도 갱신한다.

---

## 수정하지 않는 부분

- `find_sparsest_valid_cell` — 이미 Reservoir Sampling 적용됨, 변경 불필요
- `spawn_in_cell`, `try_spawn_in_cell` — 셀 기반 스폰 로직은 그대로 유지
- `generate_particle_mesh` — 이미 `spawn_in_random_triangle_global` 폴백 사용 중이므로 자동 반영
- `spawn_in_viewport_fallback` — 내부에서 위 두 함수를 호출하므로 자동 반영
- 기존 bulk generation 로직 (line 1020~1028) — 변경 불필요

---

## 검증

### 빌드
```bash
wasm-pack build --target web
```

### 시각적 확인
- 해안선 사각형 클러스터 소멸 여부
- 외해 영역 파티클 분포 존재 여부
- 만(灣) 내부 영역 파티클 존재 여부

### 콘솔 진단 (frame#0 로그)
- `none_count` < 5%
- `valid_uv` > 80%
