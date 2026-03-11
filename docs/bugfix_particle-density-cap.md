# 파티클 해류 수렴 축적 방지 — 셀 밀도 상한 도입

**생성일**: 2026-02-25  
**상태**: 구현 대기  
**관련 파일**: `lib.rs`  
**선행 조건**: 면적 가중 삼각형 선택 (bugfix_particle-area-weight) 적용 완료

---

## 문제 요약

면적 가중 스폰은 정상 동작 중(frame#0 진단: none=0, valid_uv=99.7%). 그러나 **파티클이 해류를 따라 이동하면서 수렴 지점(해안선 등)에 축적**되어 밀도가 극도로 불균형해진다.

리스폰 시점에만 균등 분포를 보장하고, 이동 중에는 밀도 제어가 없기 때문이다. life=100 프레임 동안 파티클이 스폰 셀에서 멀리 이동하여 특정 셀에 수십 개씩 쌓인다.

---

## 해결 전략: 이동 후 셀 밀도 상한 (Cell Density Cap)

파티클 이동 후, 도착 셀의 파티클 수가 상한을 초과하면 해당 파티클을 즉시 RESPAWN 마킹한다. 이렇게 하면 해류 수렴 지점의 과밀 파티클이 자동으로 희소 영역으로 재배치된다.

### 상수 추가

```rust
/// 셀당 최대 파티클 수 (목표의 2배)
/// 목표: (CELL_SIZE / TARGET_SPACING)² = (50/25)² = 4
/// 상한: 4 * 2 = 8
const MAX_PARTICLES_PER_CELL: u32 = 8;
```

위치: 기존 상수 블록 (`TARGET_SPACING`, `MIN_PARTICLES` 등) 바로 아래.

---

## 수정 지시

### `update_particles_mesh` 함수 내 파티클 루프 수정 (line 1145~1210)

현재 루프 구조는:

```
for particle in self.particles.iter_mut() {
    1. deferred respawn 처리
    2. UV 보간 → 이동
    3. should_respawn 체크 → RESPAWN 마킹
    4. 직렬화
}
```

**이동 직후 (step 2와 3 사이)에 밀도 상한 체크를 삽입한다.**

#### 수정 위치: line 1176~1193 (UV 유효 + 비제로 분기 내부)

현재 코드:

```rust
} else {
    particle.coordinate.longitude += u * exaggeration;
    particle.coordinate.latitude += v * exaggeration;
    particle.current_u = u;
    particle.current_v = v;
    particle.status = ParticleStatus::OK;

    if particle.prev_coordinats.len() >= prev_count as usize {
        particle.prev_coordinats.remove(0);
    }
    if particle.loop_count % 2 == 0 {
        particle.prev_coordinats.push(particle.coordinate);
    }
    particle.loop_count += 1;

    if should_respawn(particle) {
        particle.status = ParticleStatus::RESPAWN;
    }
}
```

수정 후:

```rust
} else {
    particle.coordinate.longitude += u * exaggeration;
    particle.coordinate.latitude += v * exaggeration;
    particle.current_u = u;
    particle.current_v = v;
    particle.status = ParticleStatus::OK;

    if particle.prev_coordinats.len() >= prev_count as usize {
        particle.prev_coordinats.remove(0);
    }
    if particle.loop_count % 2 == 0 {
        particle.prev_coordinats.push(particle.coordinate);
    }
    particle.loop_count += 1;

    if should_respawn(particle) {
        particle.status = ParticleStatus::RESPAWN;
    }

    // === 셀 밀도 상한 체크 ===
    // 이동 후 도착 셀의 파티클 수가 상한 초과 시 RESPAWN 마킹
    if particle.status != ParticleStatus::RESPAWN {
        let (sx, sy) = vp.geo_to_screen(
            particle.coordinate.longitude,
            particle.coordinate.latitude,
        );
        if sx >= 0.0 && sx < vp.canvas_w && sy >= 0.0 && sy < vp.canvas_h {
            let (col, row) = vp.screen_to_cell(sx, sy);
            let idx = row * vp.grid_cols + col;
            if idx < cell_counts.len() {
                cell_counts[idx] += 1;
                if cell_counts[idx] > MAX_PARTICLES_PER_CELL {
                    particle.status = ParticleStatus::RESPAWN;
                    cell_counts[idx] -= 1; // 카운트 롤백
                }
            }
        }
    }
}
```

#### 주의: cell_counts 초기화 방식 변경

현재 `cell_counts`는 루프 시작 전에 `build_cell_counts`로 전체 파티클 위치를 기반으로 빌드된다 (line 1103-1107). 밀도 상한 체크를 위해서는 **루프 진행 중 실시간으로 카운트를 관리**해야 한다.

**방법**: 루프 시작 전에 `cell_counts`를 0으로 초기화하고, 루프 내에서 각 파티클의 이동 후 위치를 카운트에 반영한다.

line 1103-1107 변경:

현재:
```rust
let mut cell_counts = if let Some(ref vp) = self.viewport {
    build_cell_counts(&self.particles, vp)
} else {
    Vec::new()
};
```

변경:
```rust
let mut cell_counts = if let Some(ref vp) = self.viewport {
    // 밀도 상한 체크를 위해 0으로 시작, 루프 내에서 실시간 빌드
    vec![0u32; vp.grid_cols * vp.grid_rows]
} else {
    Vec::new()
};
```

그리고 밀도 상한 체크 코드(위에서 추가한 부분)가 모든 경우(OK 이동, None 리스폰, zero_uv 리스폰)에서 카운트를 갱신하도록 한다.

#### None 분기와 zero_uv 분기에도 카운트 갱신 추가

**zero_uv 리스폰 분기** (line 1166-1174):
리스폰된 파티클의 위치에 대해서는 `spawn_in_cell`이 이미 `counts`를 갱신하므로 추가 작업 불필요.

**None 분기** (line 1196-1204):
동일하게 `spawn_in_cell`이 이미 `counts`를 갱신하므로 추가 작업 불필요.

**deferred respawn 분기** (line 1147-1154):
동일하게 `spawn_in_cell`이 이미 `counts`를 갱신하므로 추가 작업 불필요.

---

## 수정하지 않는 부분

- `spawn_in_random_triangle_viewport`, `spawn_in_random_triangle_global` — 면적 가중 수정 유지
- `find_sparsest_valid_cell` — Reservoir Sampling 유지
- `should_respawn` — 기존 로직 유지
- `adjust_particle_count_gradual` — 기존 로직 유지
- `bulk_migrate_if_needed` — 기존 로직 유지
- 레거시 경로 (viewport 없는 분기, line 1213~) — 변경 불필요

---

## 검증

### 시각적 확인
- 해안선 수렴 지점의 과밀 클러스터 소멸 여부
- 외해 희소 영역 파티클 증가 여부
- 전체적으로 균일한 밀도감 확보 여부

### 콘솔 진단
frame#0 diag 로그에서:
- `valid_uv`가 이전과 비슷한 수준 유지 (99%+)
- `none`이 여전히 0 또는 매우 낮음

### 파라미터 튜닝
`MAX_PARTICLES_PER_CELL` 값에 따른 시각적 차이:
- **6**: 더 균일하지만 수렴 패턴이 약해져 해류 시각화 효과 감소
- **8** (권장): 적절한 균형 — 수렴 패턴은 보이되 과밀 축적 방지
- **12**: 느슨하여 여전히 해안 축적 발생 가능

---

## 동작 원리 요약

```
이동 전: 셀 A=4개, 셀 B=4개 (균등)
  ↓ 해류가 A→B 방향
이동 후: 셀 A=2개, 셀 B=6개 (수렴)
  ↓ 밀도 상한 체크 (상한=8)
결과: 셀 B=6개 → 8 이하이므로 통과

더 진행:
이동 후: 셀 A=0개, 셀 B=9개 (과밀)
  ↓ 밀도 상한 체크
결과: 9번째 파티클 → RESPAWN 마킹 → 다음 프레임에 셀 A(최소 밀도)로 리스폰
  → 셀 B=8개, 셀 A=1개
```

해류 수렴 패턴은 유지하되(상한까지는 허용), 극단적 축적은 방지한다.
