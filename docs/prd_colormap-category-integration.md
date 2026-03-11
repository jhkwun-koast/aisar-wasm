# 컬러맵 카테고리 통합 UI + 팔레트 시스템

**이슈**: #442
**생성일**: 2026-03-11
**상태**: 계획 수립

---

## 배경

현재 헤더 바에 "해류(컬러맵)", "수온" 버튼이 각각 존재한다.
컬러맵 종류가 늘어날 것을 고려하여, 하나의 "컬러맵" 카테고리 버튼으로 통합하고
드롭다운에서 하위 항목을 선택하는 구조로 변경한다.

추가로, 해류 컬러맵의 색상 팔레트를 Windy / 해양조사원(KHOA) 두 가지로 전환할 수 있게 한다.

---

## 요구사항

### UI
- 기존 "해류(컬러맵)" + "수온" 버튼 → **"컬러맵 ▼"** 카테고리 버튼 1개로 통합
- 드롭다운 하위 항목: 해류(Windy), 해류(해양조사원), 수온
- 라디오 동작 (1개만 활성), 재클릭 시 OFF
- 향후 항목 추가 용이한 구조

### 범례
- 해류 컬러맵 활성 시: **해양조사원 범례 항상 표시** + Windy 선택 시 Windy 범례 추가
- 수온 컬러맵 활성 시: 수온 범례 표시

### 팔레트
- Windy: 10단계 연속 그라디언트 (남색→파랑→청록→초록→연두→노랑→주황→분홍→보라)
- 해양조사원(KHOA): 5단계 속도 구간별 색상 (기존 벡터 범례와 동일)

### 수온
- 수온 컬러맵 렌더링은 이번 범위 밖 (UI 뼈대만 준비)
- 기존 이미지 기반 수온 레이어 유지
- 향후 SCHISM 파일의 수온 데이터를 해류와 동일한 Worker+WASM 파이프라인으로 렌더링 예정

---

## 설계

### 핵심 개념: 컬러맵 모드

해류 Windy / 해류 KHOA는 **같은 레이어, 같은 데이터, 다른 팔레트**이다.
레이어를 2개 만들지 않고 팔레트 전환 메시지로 처리한다.

```
COLORMAP_MODE:
  NONE              → 모든 컬러맵 OFF
  CURRENTS_WINDY    → 해류 컬러맵 ON + Windy 팔레트
  CURRENTS_KHOA     → 해류 컬러맵 ON + KHOA 팔레트
  TEMPERATURE       → 수온 레이어 ON (기존 이미지 기반)
```

### 상태 전환 매트릭스

| 이전 모드 → 새 모드 | 동작 |
|---------------------|------|
| NONE → CURRENTS_WINDY | colormap layer ON + Windy 팔레트 설정 |
| NONE → CURRENTS_KHOA | colormap layer ON + KHOA 팔레트 설정 |
| NONE → TEMPERATURE | temperature layer ON |
| CURRENTS_WINDY → CURRENTS_KHOA | **팔레트만 전환** (레이어 유지, 데이터 재fetch 없음) |
| CURRENTS_KHOA → CURRENTS_WINDY | **팔레트만 전환** |
| CURRENTS_* → TEMPERATURE | colormap layer OFF → temperature layer ON |
| TEMPERATURE → CURRENTS_* | temperature layer OFF → colormap layer ON + 팔레트 설정 |
| 아무 모드 → NONE (재클릭) | 해당 레이어 OFF |

---

## 역할 분담: Client vs WASM

### Client (JS) 담당

| 항목 | 설명 |
|------|------|
| 드롭다운 UI | 헤더 바 "컬러맵 ▼" 버튼 + 드롭다운 메뉴 |
| 상태 관리 | `AISAR.colormapMode` — 현재 활성 모드 |
| 레이어 토글 | 모드에 따라 colormap/temperature 레이어 ON/OFF |
| 팔레트 전환 요청 | Worker에 `{ type: 'updatePalette', palette: 'windy' | 'khoa' }` 전송 |
| 범례 | 모드에 따라 범례 섹션 표시/숨김 |
| 흐름 레이어 연동 | 컬러맵 ON 시 파티클 colorMode → 'white' |
| JS 폴백 팔레트 | WASM 미사용 시 JS Worker 내 LUT 전환 |

### WASM (ColormapRenderer) 담당

| 항목 | 설명 |
|------|------|
| 팔레트 저장 | Windy / KHOA 두 팔레트를 내장 |
| 팔레트 전환 API | `set_palette(palette_name: &str)` 메서드 추가 |
| 렌더링 | `render()` 시 현재 설정된 팔레트로 RGBA 픽셀 생성 |
| LUT 관리 | 팔레트별 256단계 LUT 사전 생성 |

---

## WASM 인터페이스 변경

### 현재

```rust
#[wasm_bindgen]
impl ColormapRenderer {
    pub fn new() -> Self;
    pub fn load_mesh(&mut self, mesh_binary: &[u8]);
    pub fn load_uv(&mut self, uv_binary: &[u8], shuffled: bool);
    pub fn render(&self, extent: &[f64], width: u32, height: u32) -> Vec<u8>;
}
```

### 변경 후

```rust
#[wasm_bindgen]
impl ColormapRenderer {
    pub fn new() -> Self;
    pub fn load_mesh(&mut self, mesh_binary: &[u8]);
    pub fn load_uv(&mut self, uv_binary: &[u8], shuffled: bool);
    pub fn render(&self, extent: &[f64], width: u32, height: u32) -> Vec<u8>;

    /// 팔레트 전환 — "windy" 또는 "khoa"
    /// 전환 즉시 내부 LUT 교체, 다음 render()부터 적용
    pub fn set_palette(&mut self, name: &str);
}
```

### Windy 팔레트 (기존, 10단계 연속 보간)

```
0.00 m/s → rgb(13, 27, 74)     진한 남색
0.10     → rgb(21, 101, 192)    파랑
0.20     → rgb(0, 151, 167)     청록
0.30     → rgb(0, 137, 123)     어두운 청록
0.40     → rgb(76, 175, 80)     초록
0.60     → rgb(192, 202, 51)    연두
0.80     → rgb(255, 193, 7)     노랑
1.00     → rgb(255, 87, 34)     주황
1.20     → rgb(233, 30, 99)     분홍
1.60     → rgb(156, 39, 176)    보라
```

### KHOA 팔레트 (신규, 5단계 구간별)

기존 벡터 범례(`current-vector_legend`)와 동일한 색상 사용.
정확한 색상값과 속도 구간은 WASM 담당자가 기존 `CurrentVectorWrapper`의
내부 팔레트를 참조하여 결정.

```
(예시 — 실제 값은 CurrentVectorWrapper 내부 확인 필요)
0.00 ~ 0.25 m/s → 보라/파랑 계열
0.25 ~ 0.50     → 파랑 계열
0.50 ~ 0.75     → 초록 계열
0.75 ~ 1.00     → 주황 계열
1.00+           → 빨강 계열
```

KHOA 팔레트도 연속 보간으로 구현할지, 구간별 flat 색상으로 구현할지는
해양조사원 범례 디자인에 맞춰 결정.

---

## Client 구현 Phase

### Phase 1: 인프라 (상수 + Worker 팔레트 전환)

| 작업 | 파일 | 설명 |
|------|------|------|
| COLORMAP_MODE enum 추가 | `constant.js` | `NONE`, `CURRENTS_WINDY`, `CURRENTS_KHOA`, `TEMPERATURE` |
| updatePalette 핸들러 | `currentsColormapWorker.js` | KHOA LUT 추가, `updatePalette` 메시지로 활성 LUT 전환 (JS 폴백용) |
| setColormapPalette export | `currentsColormapLayer.js` | Worker에 팔레트 전환 메시지 전송 함수 |
| legendKey 추가 | `currentsColormapLayer.js` | `needLegend: true`, `legendKey: 'weather-colormap'` |
| elementId 제거 | `currentsColormapLayer.js`, `temperatureLayer.js` | 개별 버튼 DOM 제거에 대응 |

### Phase 2: 헤더 바 UI

| 작업 | 파일 | 설명 |
|------|------|------|
| 드롭다운 HTML | `header-layer.html` | 2개 버튼 → "컬러맵 ▼" + 드롭다운 메뉴 (radio input) |
| 드롭다운 CSS | `main.css` | 위치, 스타일, 활성 상태 |
| 모드 전환 핸들러 | `index.js` | `setColormapMode()`, 드롭다운 토글, 외부 클릭 닫기 |
| 컨테이너 너비 조정 | `header-layer.html` | 버튼 수 감소 → 너비 축소 가능 |

### Phase 3: 범례 연동

| 작업 | 파일 | 설명 |
|------|------|------|
| 컬러맵 서브탭 추가 | `legend-dialog.html` | "컬러맵" 탭 + KHOA/Windy/수온 섹션 |
| Windy 범례 캔버스 | `legend.js` | 10색 그라디언트 + 속도 라벨 생성 함수 |
| syncColormapLegend | `LegendController.js` | 모드에 따라 범례 섹션 표시/숨김 |
| temperatureLayer legendKey | `temperatureLayer.js` | `'weather-colormap'`으로 변경 |

### Phase 4: 검증

| 작업 | 설명 |
|------|------|
| 모드 전환 테스트 | Windy↔KHOA↔수온↔OFF 전환 시 레이어/범례/파티클 색상 정합성 |
| 기존 레이어 영향 없음 확인 | 바람/해류흐름/해류방향/격자망/부이 등 |
| 팔레트 전환 시 깜박임 없음 확인 | Windy→KHOA 전환 시 데이터 재fetch 없이 즉시 색상 변경 |

---

## WASM 담당 작업 목록

| 순서 | 작업 | 설명 |
|------|------|------|
| W-1 | `set_palette(&str)` 메서드 추가 | `"windy"` / `"khoa"` 문자열로 내부 LUT 전환 |
| W-2 | KHOA 팔레트 정의 | `CurrentVectorWrapper` 내부 팔레트 참조하여 동일 색상 사용 |
| W-3 | LUT 사전 생성 | Windy LUT + KHOA LUT를 `new()` 시 모두 생성, `set_palette`로 활성 LUT 포인터만 전환 |
| W-4 | 기본 팔레트 설정 | `new()` 시 기본값 = `"windy"` (현재 동작 유지) |
| W-5 | 팔레트 통일 확인 | `ColormapRenderer`와 `CurrentVectorWrapper`의 KHOA 색상이 동일한지 확인 |

### WASM-Client 연동 프로토콜

```
Main Thread → Colormap Worker:
  { type: 'updatePalette', palette: 'windy' | 'khoa' }

Colormap Worker → WASM:
  wasmRenderer.set_palette('windy')  또는  wasmRenderer.set_palette('khoa')
  → 다음 handleRender() 시 새 팔레트로 렌더링

Colormap Worker → JS 폴백:
  활성 LUT를 colorLutWindy 또는 colorLutKhoa로 전환
  → 다음 renderImageData() 시 새 LUT 사용
```

---

## 헤더 바 변경 전후

### 변경 전 (현재)
```
[바람(흐름)] [바람(방향)] [해류(흐름)] [해류(방향)] [해류(컬러맵)] [수온] [격자망] [부이] [...]
```

### 변경 후
```
[바람(흐름)] [바람(방향)] [해류(흐름)] [해류(방향)] [컬러맵 ▼] [격자망] [부이] [...]
                                                      │
                                                      ├─ ● 해류 (Windy)
                                                      ├─ ○ 해류 (해양조사원)
                                                      └─ ○ 수온
```

- 버튼 수: 9개 → 8개 (헤더 바 여유 확보)
- 수온 버튼 제거 → 컬러맵 드롭다운 내로 이동
- 기존 수온 레이어는 그대로 유지, 토글 경로만 변경

---

## 범례 표시 규칙

| 활성 모드 | KHOA 범례 | Windy 범례 | 수온 범례 |
|-----------|----------|------------|----------|
| CURRENTS_WINDY | 표시 | 표시 | 숨김 |
| CURRENTS_KHOA | 표시 | 숨김 | 숨김 |
| TEMPERATURE | 숨김 | 숨김 | 표시 |
| NONE | 숨김 | 숨김 | 숨김 |

---

## 향후 확장

| 항목 | 설명 |
|------|------|
| 수온 컬러맵 (SCHISM) | SCHISM 파일의 수온 데이터를 해류와 동일한 Worker+WASM 파이프라인으로 렌더링 |
| 파고 컬러맵 | 드롭다운에 항목 추가 + 새 팔레트/데이터 소스 연결 |
| 사용자 팔레트 커스터마이징 | 팔레트 편집 UI (장기) |

---

## 리스크

| 리스크 | 등급 | 대응 |
|--------|------|------|
| elementId 제거 시 기존 참조 깨짐 | 낮 | `BasicLayer.syncElementVisible()`이 null 체크함 |
| 팔레트 전환 시 깜박임 | 낮 | 이전 bitmap clear + 즉시 재렌더 |
| WASM set_palette 미구현 시 | 중 | JS 폴백 LUT 전환으로 동작 보장 |
| 더보기 드롭다운과 충돌 | 낮 | 외부 클릭 핸들러 통합 |
