# SCHISM 파티클 WASM 성능 개선 — PRNG 교체 + 버퍼 재사용

**생성일**: 2026-02-25
**상태**: 계획 수립 중
**담당**: WASM (Rust) 개발
**우선순위**: 긴급 — 이 작업 완료 후 Client 측 Worker 오프로딩 진행

---

## 1. 현재 성능 문제

Chrome DevTools 프로파일링(58.4초 기록, 110,559 샘플) 결과:

```
#1  crypto.getRandomValues()    22.1초  37.8%  ← 이 작업에서 해결
#2  WASM 함수 (파티클 계산)     12.7초  21.8%
#3  GC (가비지 컬렉션)           7.3초  12.5%  ← 이 작업에서 개선
#4  JS WASM glue (배열 복사)     7.2초  12.4%  ← 이 작업에서 개선
#5  WASM↔JS 경계 횡단            2.9초   5.0%
```

해류 레이어 활성화 시 **첫 프레임이 43.4초** 동안 메인 스레드를 점유하여 UI가 완전히 멈춘다.

---

## 2. 작업 1: PRNG 교체 (CPU 38% 제거)

### 원인

Rust `rand` 크레이트가 WASM(wasm32-unknown-unknown) 타겟에서 `thread_rng()` → `OsRng` → `getrandom` → JS `crypto.getRandomValues()`를 사용한다. 모든 랜덤 호출이 WASM→JS 경계를 넘어 암호학적 난수 생성기를 거치는 것이다.

파티클 시뮬레이션(스폰 위치, 수명 랜덤화, 삼각형 내 포인트 선택)에 암호학적 난수는 불필요하다.

### 해결

구조체에 PRNG 인스턴스를 필드로 보관하고, 초기화 시 한 번만 `OsRng`로 시드를 생성한 뒤, 이후 모든 랜덤 호출은 산술 PRNG를 사용한다.

**Cargo.toml 의존성 추가**:

```toml
[dependencies]
rand = "0.8"
rand_xoshiro = "0.6"
# getrandom은 기존대로 js feature 유지 (초기 시드 1회용)
getrandom = { version = "0.2", features = ["js"] }
```

**변경 패턴**:

```rust
use rand::SeedableRng;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct CurrentFlowWrapper {
    // 기존 필드들...
    rng: Xoshiro256PlusPlus,  // 추가
}

impl CurrentFlowWrapper {
    pub fn new(/* ... */) -> Self {
        // OsRng로 시드 1회만 생성 (crypto.getRandomValues 1회 호출)
        let rng = Xoshiro256PlusPlus::from_rng(rand::rngs::OsRng).unwrap();
        
        Self {
            // ...
            rng,
        }
    }
    
    fn spawn_particle(&mut self) {
        // 변경 전: rand::thread_rng().gen_range(0.0..1.0)
        // 변경 후:
        let x = self.rng.gen_range(0.0..1.0);
        let y = self.rng.gen_range(0.0..1.0);
        // ...
    }
}
```

**수정 범위**: `rand::thread_rng()`, `OsRng`, `rand::random()` 를 사용하는 모든 곳을 `self.rng`로 교체한다. 주요 위치:

- 파티클 스폰 위치 결정 (삼각형 선택 + 삼각형 내 랜덤 포인트)
- 파티클 수명 초기화 (life ± 랜덤 변동)
- 파티클 리스폰 시점

### 검증

변경 전후 Chrome DevTools Performance 탭에서 `crypto.getRandomValues` 호출이 사라지는지 확인한다.

---

## 3. 작업 2: getUpdateParticles 버퍼 재사용 (GC 압력 감소)

### 원인

현재 `getUpdateParticles()`가 매 프레임 **새 Uint8Array를 할당**하여 JS로 반환한다. JS 측에서 즉시 읽고 버리므로 MinorGC가 783회(5.8초) 발생한다.

프로파일에서 관련 함수들:

```
getArrayU8FromWasm0:     2.7%   ← 배열 생성
subarray:                2.5%   ← 슬라이스
prototypesetcall:        2.3%   ← 복사
new_with_length:         0.1%   ← 할당
```

### 해결: 고정 버퍼 재사용

WASM 메모리에 고정 크기 출력 버퍼를 미리 할당하고, 매 프레임 해당 영역만 덮어쓴 뒤 JS 측에 포인터+길이를 반환한다.

**방법 A — WASM 내부 고정 버퍼 (권장)**:

```rust
pub struct CurrentFlowWrapper {
    // 기존 필드들...
    output_buffer: Vec<u8>,  // 고정 크기 출력 버퍼
}

impl CurrentFlowWrapper {
    pub fn new(/* ... */) -> Self {
        let particle_count = 3000;
        let particle_bytes = 40;
        let output_buffer = vec![0u8; particle_count * particle_bytes];
        // ...
    }
    
    /// 기존 getUpdateParticles()는 유지하되 내부 구현 변경
    pub fn getUpdateParticles(&mut self) -> &[u8] {
        // 파티클 업데이트 로직 (기존과 동일)
        // ...
        
        // output_buffer에 직접 쓰기
        for (i, particle) in self.particles.iter().enumerate() {
            let offset = i * 40;
            self.output_buffer[offset..offset+8].copy_from_slice(&particle.prev_x.to_le_bytes());
            self.output_buffer[offset+8..offset+16].copy_from_slice(&particle.prev_y.to_le_bytes());
            // ... currX, currY, speed, lifeRatio
        }
        
        &self.output_buffer[..self.particles.len() * 40]
    }
}
```

wasm-bindgen이 `&[u8]` 반환 시 JS 측에 Uint8Array 뷰를 생성하는데, 이 뷰가 WASM 메모리를 직접 참조하므로 **복사가 1회로 줄거나 0회**가 된다.

**주의사항**: `&[u8]`을 반환하면 JS 측이 다음 WASM 호출 전까지만 유효한 뷰를 받는다. 현재 JS 코드(`renderFrame`)가 같은 틱 내에서 동기적으로 소비하므로 문제 없다.

### 대안: JS 측에서 처리

WASM 수정 없이 JS 측에서도 개선 가능하다. 이 경우 Client 설계서에서 다룬다:

```javascript
// JS에서 한 번 할당 후 재사용
const reusableBuffer = new Uint8Array(MAX_PARTICLES * PARTICLE_BYTES);
```

---

## 4. 기존 API 인터페이스 (변경 없음)

Client 측 Worker 오프로딩 작업에서 WASM API 호출 방식은 변경하지 않는다. 아래 인터페이스를 그대로 유지한다.

```
CurrentFlowWrapper(binary_data: Uint8Array)  → 생성자
load_mesh_binary(data: Uint8Array)           → 메시 로드
update_uv_binary(data: Uint8Array)           → UV 갱신
set_viewport(w: number, h: number, extent: Float64Array) → 뷰포트 설정
getUpdateParticles() → Uint8Array            → 파티클 데이터 (매 프레임)
get_node_count() → number                    → 노드 수
has_mesh() → boolean                         → 메시 로드 여부
adjustParticleCount(n: number)               → 파티클 수 조정
```

### WASM 출력 포맷 (파티클당 40바이트, Little Endian)

```
offset  0: prevX     (f64) — 이전 프레임 캔버스 X
offset  8: prevY     (f64) — 이전 프레임 캔버스 Y
offset 16: currX     (f64) — 현재 프레임 캔버스 X
offset 24: currY     (f64) — 현재 프레임 캔버스 Y
offset 32: speed     (f32) — 유속 (m/s)
offset 36: lifeRatio (f32) — 수명 비율 (0.0→1.0)
```

---

## 5. 빌드 및 배포

```bash
# 빌드
wasm-pack build --target web --release

# 출력 파일 복사
cp pkg/aisar_wasm.js    → static/js/ko/wasm/aisar_wasm.js
cp pkg/aisar_wasm_bg.wasm → static/js/ko/wasm/aisar_wasm_bg.wasm
```

WASM 빌드 산출물을 교체하면 JS 측 코드 변경 없이 성능 개선이 적용된다.

---

## 6. 완료 기준

- [ ] `crypto.getRandomValues` 호출이 WASM 초기화 시 1회로 감소 (프로파일로 확인)
- [ ] 기존 파티클 동작과 시각적으로 동일 (랜덤 분포 품질 차이 없음)
- [ ] 기존 JS API 인터페이스 변경 없음 (`getUpdateParticles()` 반환 타입 동일)
- [ ] (선택) GC MinorGC 횟수 50% 이상 감소
