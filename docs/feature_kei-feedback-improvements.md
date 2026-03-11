# SCHISM 해류 가시화 개선 — KEI 피드백 반영

**생성일**: 2026-02-28
**상태**: 계획 수립 중
**관련 파일**: `currentsFlowLayer.js`, `currentsFlowWorker.js`, `currentsFlowRenderFunc.js`, `SchismDataManager.js`

---

## 피드백 요약

| # | 내용 | 우선순위 |
|---|------|---------|
| 1 | 해류 컬러맵 레이어 신규 개발 (Windy 스타일 배경 속도 채움) | **1순위** |
| 2 | 줌 레벨별 파티클 속도 불연속 해결 (exaggeration 선형화) | 2순위 |
| 3 | 파티클 굵기↑ + 갯수↓ (가시성 개선) | 3순위 |

---

## 1. 해류 컬러맵 레이어 (1순위)

### 1.1 개요

해류 속도를 삼각형 메시 기반 면적 채움(컬러맵)으로 표현하는 독립 레이어를 추가한다.
Windy.com의 "해류" 레이어와 동일한 시각적 표현을 목표로 한다.

```
기존 UI 버튼:
  [해류(흐름)] [해류(방향)]

추가 후:
  [해류(흐름)] [해류(방향)] [해류(컬러맵)]   ← 신규
```

### 1.2 두 가지 모드

| 모드 | 해류(컬러맵) | 해류(흐름) | 파티클 색상 |
|------|------------|-----------|-----------|
| A. 컬러맵 OFF + 흐름 ON | OFF | ON | **속도별 컬러** (현재) |
| B. 컬러맵 ON + 흐름 ON | ON (배경) | ON | **흰색/밝은 단색** (배경 대비) |
| C. 컬러맵 ON + 흐름 OFF | ON | OFF | — |
| D. 둘 다 OFF | OFF | OFF | — |

모드 B가 Windy 스타일의 완성형이다.
각 레이어는 독립 토글이므로 모든 조합이 자연스럽게 동작해야 한다.

### 1.3 파티클 색상 전환

해류(컬러맵) 레이어의 visible 상태를 해류(흐름) 레이어가 참조한다.

```
컬러맵 ON  → 파티클 색상: 흰색 (rgba(255,255,255,alpha))
컬러맵 OFF → 파티클 색상: 속도별 컬러 (현재 speedToColor)
```

구현 위치:
- `currentsFlowRenderFunc.js`의 `speedToColor` 호출부에 분기 추가
- Worker 모드: Worker에 `particleColorMode: 'white' | 'speed'` 메시지 전달
- 폴백 모드: `renderFrame`에 `colorMode` 파라미터 추가

### 1.4 데이터 흐름

```
SchismDataManager (공유 캐시, 싱글턴)
  ├── ensureMesh(date) → mesh ArrayBuffer (캐시)
  └── fetchUv(date, time) → UV ArrayBuffer (캐시)
        │
        ├── 해류(흐름) Worker → 독자 fetch (Worker 스레드)
        │     └ Worker는 자체 fetch, SchismDataManager 미사용
        │
        ├── 해류(흐름) 폴백 → SchismDataManager 사용
        │
        └── 해류(컬러맵) → SchismDataManager 사용 (항상 메인 스레드)
```

해류(흐름)이 Worker 모드일 때 둘 다 켜면 UV가 2회 fetch될 수 있으나,
mesh는 서버 캐시 + 브라우저 캐시(Cache-Control)로 중복 최소화되고,
UV는 ~800KB이므로 허용 범위이다.

### 1.5 레이어 구조

```javascript
// currentsColormapLayer.js — 신규 파일

const olLayer = new ol.layer.Image({
  opacity: 0.7,         // 배경이므로 약간 투명
  visible: false,
  zIndex: -1,           // 해류(흐름) 아래에 깔림
  source: new ol.source.ImageCanvas({
    canvasFunction: colormapCanvasFunction,
    ratio: 1,
  }),
});
```

핵심: **애니메이션 루프 없음** (`onPostCompose` 불필요).
시간 스텝 변경 또는 뷰포트 변경 시에만 `olLayer.getSource().changed()` 호출.

### 1.6 컬러맵 렌더링 로직

`colormapCanvasFunction(extent, resolution, pixelRatio, size)`:

```
1. SchismDataManager에서 mesh + UV 가져오기
2. mesh 바이너리 파싱 → 노드 좌표[], 삼각형 인덱스[]
3. UV 바이너리 파싱 → 노드별 (u, v)
4. 각 삼각형에 대해:
   a. 세 꼭짓점의 좌표가 뷰포트 AABB와 겹치는지 확인 (빠른 컬링)
   b. 세 꼭짓점의 speed = sqrt(u²+v²) 계산
   c. 삼각형 평균 speed = (s0+s1+s2) / 3
   d. speed → color 변환 (Windy 팔레트)
   e. 좌표를 화면 픽셀로 변환
   f. Canvas fillPath로 삼각형 채우기
```

### 1.7 JS 측 mesh/UV 파싱

SchismDataManager가 보유한 ArrayBuffer를 JS에서 직접 파싱한다.
(WASM은 파티클 레이어 전용이므로 컬러맵은 순수 JS로 처리)

```javascript
// mesh 파싱 (헤더 16B + nodes N×16B + triangles T×12B)
function parseMesh(buffer) {
  const header = new DataView(buffer, 0, 16);
  const nodeCount = header.getUint32(0, true);
  const faceCount = header.getUint32(4, true);

  const nodeOffset = 16;
  const nodes = new Float64Array(buffer, nodeOffset, nodeCount * 2);

  const triOffset = nodeOffset + nodeCount * 16;
  const triangles = new Uint32Array(buffer, triOffset, faceCount * 3);

  return { nodeCount, faceCount, nodes, triangles };
}

// UV 파싱 (Int16 양자화 포맷: 헤더 8B + N×4B)
function parseUv(buffer) {
  const header = new DataView(buffer, 0, 8);
  const scale = header.getFloat32(0, true);
  const nodeCount = header.getUint32(4, true);

  const raw = new Int16Array(buffer, 8);
  const speeds = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const u = raw[i * 2] / scale;
    const v = raw[i * 2 + 1] / scale;
    speeds[i] = Math.sqrt(u * u + v * v);
  }
  return speeds;
}
```

### 1.8 성능 고려 — 하이브리드 렌더링

삼각형 수에 따라 렌더링 방식을 자동 전환한다.

**방식 A: Canvas Path (줌 인)**
삼각형 개수가 적을 때 사용. 삼각형별로 moveTo→lineTo→lineTo→fill.
속도 버킷별로 같은 색상의 삼각형을 한 Path에 모아 fill 1회 호출로 최적화.

**방식 B: ImageData 그리드 샘플링 (줌 아웃)**
화면 전체를 4px 간격 그리드로 분할하고, 각 격자점이 어느 삼각형에 속하는지 조회하여
해당 삼각형의 속도 → 색상을 픽셀 버퍼(ImageData)에 직접 기록.
1920×1080 화면 기준 480×270 = ~13만 포인트만 처리하면 된다.

삼각형 탐색은 JS에서 간이 공간 인덱스(뷰포트를 NxM 셀로 분할, 삼각형을 셀에 할당)로 처리.

**자동 전환 임계값**: 뷰포트 내 삼각형 수 10만 개 기준

| 줌 레벨 | 뷰포트 내 삼각형 | 방식 | 예상 시간 |
|---------|---------------|------|----------|
| 줌 인 (항만) | ~5,000 | Canvas Path | **~10ms** |
| 중간 (해역) | ~50,000 | Canvas Path | **~100ms** |
| 줌 아웃 (전국) | ~490,000 | ImageData 그리드 | **~50ms** |

어느 경우든 시간 스텝 변경 시 1회만 렌더링되고, 파티클 애니메이션과 독립이므로 UI 블로킹 없음.

### 1.9 Windy 컬러 팔레트

Windy.com 해류 컬러맵 실제 범례 기준 (비선형 스케일):

```
속도 (m/s)    색상                     비고
─────────────────────────────────────────────
0.00          #0D1B4A  (진한 남색)      정지
0.10          #1565C0  (파랑)
0.20          #0097A7  (시안)           약한 해류
0.30          #00897B  (틸)
0.40          #4CAF50  (초록)           보통 해류
0.60          #C0CA33  (라임)
0.80          #FFC107  (노랑)           강한 해류
1.00          #FF5722  (주황→빨강)
1.20          #E91E63  (핑크)           매우 강한 해류
1.60+         #9C27B0  (마젠타)
```

범례 스케일 앵커: [0, 0.2, 0.4, 0.8, 1.0, 1.6] — 비선형 간격.
남색 → 시안 → 초록 → 노랑 → 빨강 → 마젠타 흐름.
구간 내는 선형 보간으로 부드러운 그라데이션 생성.

이 팔레트는 컬러맵 전용이다. 파티클이 흰색으로 전환될 때 이 배경 위에서 잘 보인다.
실제 구현 시 Windy 화면과 나란히 놓고 색상을 미세 조정한다.

### 1.10 파일 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `currentsColormapLayer.js` | **신규** | OL 레이어 + canvasFunction + 렌더링 |
| `currentsColormapRenderFunc.js` | **신규** | 삼각형 채우기 + 팔레트 + mesh/UV 파싱 |
| `SchismDataManager.js` | 수정 | UV 포맷 자동 감지를 위한 파싱 헬퍼 추가 |
| `currentsFlowLayer.js` | 수정 | 컬러맵 visible 상태 참조 → Worker에 colorMode 전달 |
| `currentsFlowWorker.js` | 수정 | colorMode 메시지 처리 → 렌더링 시 흰색/컬러 분기 |
| `currentsFlowRenderFunc.js` | 수정 | colorMode 파라미터 추가, 흰색 모드 분기 |
| `constant.js` | 수정 | `WEATHER_OCEAN_LAYER.CURRENT_COLORMAP` 추가 |
| HTML/UI | 수정 | 버튼 추가 |

---

## 2. exaggeration 선형화 (2순위)

### 2.1 현재 문제

줌 tier 경계(80km, 400km)에서 exaggeration 값이 계단식으로 변경:

```
extent 폭   exaggeration
79,999m  →  150          ┐
80,001m  →  600          ┘ ← 4배 점프! 파티클 속도 갑자기 변함
399,999m →  600          ┐
400,001m →  2000         ┘ ← 3.3배 점프!
```

### 2.2 해결: 연속 보간

tier 경계값을 앵커 포인트로 사용하되, 그 사이를 **로그 선형 보간**한다.

```
앵커 포인트:
  extent 80,000m   → exaggeration 150
  extent 400,000m  → exaggeration 600
  extent 2,000,000m → exaggeration 2000

80,000 이하  → 150 고정
2,000,000 이상 → 2000 고정
그 사이     → 로그 스케일 선형 보간
```

로그 스케일을 사용하는 이유: extent가 80km→400km→2000km로 **지수적으로** 변하므로,
선형 보간하면 저줌에서 변화가 너무 급하고 고줌에서 너무 완만해진다.

```javascript
const ANCHORS = [
  { extent: 80000,   exag: 150 },
  { extent: 400000,  exag: 600 },
  { extent: 2000000, exag: 2000 },
];

function getExaggeration(extentWidth) {
  if (extentWidth <= ANCHORS[0].extent) return ANCHORS[0].exag;
  if (extentWidth >= ANCHORS[ANCHORS.length - 1].extent)
    return ANCHORS[ANCHORS.length - 1].exag;

  // 해당 구간 찾기
  for (let i = 0; i < ANCHORS.length - 1; i++) {
    if (extentWidth <= ANCHORS[i + 1].extent) {
      // 로그 스케일 보간
      const logE = Math.log(extentWidth);
      const logA = Math.log(ANCHORS[i].extent);
      const logB = Math.log(ANCHORS[i + 1].extent);
      const t = (logE - logA) / (logB - logA);
      return ANCHORS[i].exag + (ANCHORS[i + 1].exag - ANCHORS[i].exag) * t;
    }
  }
}
```

### 2.3 변경 범위

| 파일 | 변경 |
|------|------|
| `currentsFlowLayer.js` | `ZOOM_TIERS` 배열 → `ANCHORS` + `getExaggeration()` 함수 |
| | `getZoomTier()` → 뷰포트 변경마다 연속 exaggeration 계산 |
| | tier 비교(`tier !== currentZoomTier`) → 값 변경 감지로 전환 |
| `currentsFlowWorker.js` | `handleUpdateParams`가 `set_exaggeration(value)` 호출 (이미 설계됨) |

WASM의 `set_exaggeration` 메서드는 이전 Phase에서 이미 구현 준비 완료.
wrapper 재생성 없이 값만 변경하므로 메시/UV 리로드 없음.

### 2.4 검증 방법

줌 슬라이더를 천천히 돌리면서:
- 파티클 속도가 급변하는 지점이 없는지 확인
- 줌 인/아웃 방향 모두에서 부드러운 전환 확인
- 최대 줌/최소 줌에서 속도가 적절한지 확인

---

## 3. 파티클 가시성 개선 (3순위)

### 3.1 변경 내용

| 파라미터 | 현재 | 변경 | 파일 |
|---------|------|------|------|
| LINE_WIDTH | 2.5 | **3.0~3.5** | `currentsFlowRenderFunc.js` |
| 파티클 밀도 | `canvasArea / 625` | `canvasArea / 900~1200` | WASM `lib.rs` |

LINE_WIDTH를 올리면 개별 파티클이 더 잘 보이고,
갯수를 줄이면 파티클 간 간격이 넓어져서 개별 흐름선이 구분된다.

### 3.2 주의사항

- LINE_WIDTH는 Worker 모드/폴백 모드 양쪽에 적용 필요
  - Worker: `currentsFlowWorker.js` 내 렌더링 코드
  - 폴백: `currentsFlowRenderFunc.js`
- 파티클 수 변경은 WASM(`lib.rs`)의 `calculate_particle_count()` 수정
  - 분모값 조정: 625(현재 25px 간격) → 900(30px) 또는 1225(35px)
- 컬러맵 ON 시 흰색 파티클은 굵기가 더 중요 (배경 위 대비)
  - 컬러맵 모드에서 LINE_WIDTH를 추가로 0.5px 올리는 것도 검토

### 3.3 파라미터 튜닝 전략

정확한 값은 실제 화면을 보며 조정해야 하므로, 상수를 한 곳에 모아서 빠르게 변경 가능하도록 한다:

```javascript
// currentsFlowRenderFunc.js 상단
const STYLE = {
  LINE_WIDTH_SPEED: 3.0,      // 속도 컬러 모드
  LINE_WIDTH_WHITE: 3.5,      // 흰색 모드 (컬러맵 배경 위)
  MIN_SEG_LEN: 6,
  MAX_SEG_LEN: 20,
};
```

---

## 구현 순서

### Phase 1: 컬러맵 레이어 신규 개발

| 순서 | 작업 |
|------|------|
| 1-1 | `currentsColormapRenderFunc.js` — mesh/UV 파싱 + 삼각형 채우기 + 팔레트 |
| 1-2 | `currentsColormapLayer.js` — OL 레이어 + SchismDataManager 연동 |
| 1-3 | `constant.js`, HTML — 버튼 추가, 레이어 등록 |
| 1-4 | 파티클 흰색 전환 — `currentsFlowRenderFunc.js`, Worker에 colorMode 전달 |
| 1-5 | 검증: 컬러맵 단독 / 흐름 단독 / 둘 다 켜기 / 시간 변경 |

### Phase 2: exaggeration 선형화

| 순서 | 작업 |
|------|------|
| 2-1 | WASM `set_exaggeration()` 메서드 추가 (lib.rs 1줄) |
| 2-2 | `currentsFlowLayer.js` — ANCHORS + getExaggeration() |
| 2-3 | `currentsFlowWorker.js` — handleUpdateParams 단순화 |
| 2-4 | 검증: 줌 슬라이더 천천히 조작하며 부드러운 전환 확인 |

### Phase 3: 파티클 가시성 조정

| 순서 | 작업 |
|------|------|
| 3-1 | LINE_WIDTH 상수 변경 (renderFunc + Worker) |
| 3-2 | WASM 파티클 밀도 분모 조정 |
| 3-3 | 컬러맵 모드 흰색 LINE_WIDTH 별도 설정 |
| 3-4 | 검증: Windy와 비교하며 파라미터 조정 |
