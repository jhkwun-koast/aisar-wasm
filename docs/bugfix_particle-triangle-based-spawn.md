# 셀 기반 스폰 → 삼각형 기반 스폰 전환

**생성일**: 2026-02-25  
**상태**: 구현 대기  
**관련 파일**: `lib.rs`  
**선행 조건**: 면적 가중 삼각형 선택 적용 완료

---

## 문제 요약

`find_sparsest_valid_cell` → `spawn_in_cell` → `try_spawn_in_cell` 흐름이 50×50px 셀 내부에서 랜덤 좌표를 생성한다. 해안 근처 셀은 물 영역이 셀 구석에만 있어서 파티클이 사각형 격자 패턴으로 집중된다.

```
셀 경계 (50×50px)
┌──────────┐
│ 육지 육지 │
│ 육지 ··· │ ← 구석의 물 영역에만 파티클 집중
│ 육지 ··· │
└──────────┘
```

---

## 해결: 삼각형 기반 스폰 + 밀도 rejection

스폰을 항상 면적 가중 삼각형 선택으로 수행하고, 스폰 후 도착 셀의 밀도가 상한을 초과하면 재시도한다.

### 새 함수: `spawn_balanced_particle`

이 함수가 모든 리스폰 경로를 대체한다.

```rust
/// 면적 가중 삼각형 스폰 + 밀도 rejection
/// 최대 max_attempts회 시도, 실패 시 마지막 시도 결과를 그대로 반환
fn spawn_balanced_particle(
    viewport: &Viewport,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    life: i16,
    counts: &mut Vec<u32>,
    max_per_cell: u32,
) -> Option<Particle> {
    const MAX_ATTEMPTS: usize = 5;

    let mut last_particle: Option<Particle> = None;

    for _ in 0..MAX_ATTEMPTS {
        // 면적 가중 삼각형 선택 → 삼각형 내부 랜덤 좌표
        let p = spawn_in_random_triangle_viewport(viewport, rtree, nodes, life, counts);
        let p = match p {
            Some(p) => p,
            None => break,
        };

        let (sx, sy) = viewport.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
        if sx >= 0.0 && sx < viewport.canvas_w && sy >= 0.0 && sy < viewport.canvas_h {
            let (col, row) = viewport.screen_to_cell(sx, sy);
            let idx = row * viewport.grid_cols + col;
            if idx < counts.len() && counts[idx] < max_per_cell {
                // 밀도 OK → 채택
                counts[idx] += 1;
                return Some(p);
            }
        }

        // 밀도 초과 → last_particle에 저장하고 재시도
        last_particle = Some(p);
    }

    // 모든 시도가 밀도 초과 → 마지막 결과를 그대로 사용 (데드락 방지)
    if let Some(ref p) = last_particle {
        let (sx, sy) = viewport.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
        if sx >= 0.0 && sx < viewport.canvas_w && sy >= 0.0 && sy < viewport.canvas_h {
            let (col, row) = viewport.screen_to_cell(sx, sy);
            let idx = row * viewport.grid_cols + col;
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }
    }
    last_particle
}
```

위치: `spawn_in_viewport_fallback` 바로 아래에 추가.

**주의**: `spawn_in_random_triangle_viewport` 함수가 내부에서 `counts`를 갱신하는 부분이 있다 (line 2280-2288). `spawn_balanced_particle`에서 rejection할 때 이중 카운트가 발생하지 않도록, `spawn_in_random_triangle_viewport`의 카운트 갱신 부분을 **제거**하고, 카운트 관리를 `spawn_balanced_particle`에서만 처리한다.

### `spawn_in_random_triangle_viewport` 수정

line 2280-2288의 셀 카운트 업데이트 블록을 제거:

```rust
// 아래 블록 삭제
    // 셀 카운트 업데이트
    let (sx, sy) = viewport.geo_to_screen(x, y);
    if sx >= 0.0 && sx < viewport.canvas_w && sy >= 0.0 && sy < viewport.canvas_h {
        let (col, row) = viewport.screen_to_cell(sx, sy);
        let idx = row * viewport.grid_cols + col;
        if idx < counts.len() {
            counts[idx] += 1;
        }
    }
```

그리고 함수 시그니처에서 `counts` 파라미터를 제거:

```rust
fn spawn_in_random_triangle_viewport(
    viewport: &Viewport,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    life: i16,
    // counts 파라미터 제거
) -> Option<Particle> {
```

이에 따라 `spawn_in_cell` (line 2379)의 폴백 호출도 수정:
```rust
// 기존
spawn_in_random_triangle_viewport(viewport, rtree, nodes, life, counts)
// 변경
spawn_in_random_triangle_viewport(viewport, rtree, nodes, life)
```

---

## 호출부 교체

### 1. `update_particles_mesh` 내 파티클 루프 (line 1145~1210)

**deferred respawn** (line 1147-1154):
```rust
// 기존
if particle.status == ParticleStatus::RESPAWN {
    let respawned = find_sparsest_valid_cell(&cell_counts, mesh_mask, vp.grid_cols, vp.grid_rows)
        .and_then(|(col, row)| spawn_in_cell(col, row, vp, rtree, nodes, life, &mut cell_counts));
    if let Some(new_p) = respawned {
        *particle = new_p;
    } else {
        *particle = spawn_in_viewport_fallback(vp, rtree, nodes, mesh_bounds, life, &mut cell_counts);
    }
}

// 변경
if particle.status == ParticleStatus::RESPAWN {
    if let Some(new_p) = spawn_balanced_particle(vp, rtree, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL) {
        *particle = new_p;
    }
    // None이면 그대로 둠 (극히 드문 경우, 다음 프레임에 재시도)
}
```

**zero_uv 리스폰** (line 1166-1174):
```rust
// 기존
if u.abs() < EPSILON && v.abs() < EPSILON {
    let respawned = find_sparsest_valid_cell(&cell_counts, mesh_mask, vp.grid_cols, vp.grid_rows)
        .and_then(|(c, r)| spawn_in_cell(c, r, vp, rtree, nodes, life, &mut cell_counts));
    if let Some(new_p) = respawned {
        *particle = new_p;
    } else {
        *particle = spawn_in_viewport_fallback(vp, rtree, nodes, mesh_bounds, life, &mut cell_counts);
    }
}

// 변경
if u.abs() < EPSILON && v.abs() < EPSILON {
    if let Some(new_p) = spawn_balanced_particle(vp, rtree, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL) {
        *particle = new_p;
    }
}
```

**None(메시 밖) 리스폰** (line 1196-1204):
```rust
// 기존
None => {
    let respawned = find_sparsest_valid_cell(&cell_counts, mesh_mask, vp.grid_cols, vp.grid_rows)
        .and_then(|(c, r)| spawn_in_cell(c, r, vp, rtree, nodes, life, &mut cell_counts));
    if let Some(new_p) = respawned {
        *particle = new_p;
    } else {
        *particle = spawn_in_viewport_fallback(vp, rtree, nodes, mesh_bounds, life, &mut cell_counts);
    }
}

// 변경
None => {
    if let Some(new_p) = spawn_balanced_particle(vp, rtree, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL) {
        *particle = new_p;
    }
}
```

### 2. `adjust_particle_count_gradual` (line 933-938):
```rust
// 기존
for _ in 0..to_add {
    if let Some((col, row)) = find_sparsest_valid_cell(&cell_counts, mesh_mask, viewport.grid_cols, viewport.grid_rows) {
        if let Some(p) = spawn_in_cell(col, row, &viewport, rtree, nodes, life, &mut cell_counts) {
            self.particles.push(p);
        }
    }
}

// 변경
for _ in 0..to_add {
    if let Some(p) = spawn_balanced_particle(&viewport, rtree, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL) {
        self.particles.push(p);
    }
}
```

### 3. `migrate_particles_to_viewport` (line 976-981):
```rust
// 기존
for &idx in out_indices.iter().take(to_migrate) {
    if let Some((col, row)) = find_sparsest_valid_cell(&cell_counts, mesh_mask, viewport.grid_cols, viewport.grid_rows) {
        if let Some(new_p) = spawn_in_cell(col, row, &viewport, rtree, nodes, life, &mut cell_counts) {
            self.particles[idx] = new_p;
        }
    }
}

// 변경
for &idx in out_indices.iter().take(to_migrate) {
    if let Some(new_p) = spawn_balanced_particle(&viewport, rtree, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL) {
        self.particles[idx] = new_p;
    }
}
```

### 4. `bulk_migrate_if_needed` (line 1021-1028):
```rust
// 기존
for _ in 0..target {
    if let Some((col, row)) = find_sparsest_valid_cell(&cell_counts, &mask, vp.grid_cols, vp.grid_rows) {
        if let Some(p) = spawn_in_cell(col, row, &vp, rtree, nodes, life, &mut cell_counts) {
            particles.push(p);
            continue;
        }
    }
    particles.push(generate_particle_mesh(mesh_bounds, rtree, nodes, life));
}

// 변경
for _ in 0..target {
    if let Some(p) = spawn_balanced_particle(&vp, rtree, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL) {
        particles.push(p);
    } else {
        // 극단적 폴백
        particles.push(generate_particle_mesh(mesh_bounds, rtree, nodes, life));
    }
}
```

---

## 이동 중 밀도 상한 체크 (density cap)

이전 bugfix_particle-density-cap.md에서 추가한 이동 후 밀도 상한 체크는 **유지**한다. 스폰 시 밀도 제어 + 이동 후 밀도 제어의 이중 방어.

---

## 더 이상 불필요한 함수들

아래 함수들은 더 이상 호출되지 않지만, 삭제하지 않고 그대로 둔다 (향후 참조용).

- `find_sparsest_valid_cell` — 사용처 없음
- `spawn_in_cell` — `spawn_in_random_triangle_viewport` 내부 폴백에서만 참조
- `try_spawn_in_cell` — `spawn_in_cell`에서만 참조
- `spawn_in_viewport_fallback` — 사용처 없음
- `build_mesh_cell_mask` — `adjust_particle_count_gradual`의 valid_cells 계산에서 여전히 사용

---

## 검증

### 시각적 확인
- **핵심**: 해안 근처 사각형 격자 패턴 소멸 여부
- 외해 파티클 분포 균일 여부
- 해안 근처에서도 자연스러운(비격자) 분포 여부

### 콘솔
- `frame#0 diag`에서 `none=0` 유지 확인
- `valid_uv` 90%+ 유지 확인
