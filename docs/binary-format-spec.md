# SCHISM Binary API Format Specification

SCHISM 해류 시각화를 위한 바이너리 데이터 포맷 명세.
WebAssembly 파티클 엔진에서 이 포맷을 파싱하여 GPU 렌더링에 사용한다.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/schism/mesh?date={yyyyMMdd}` | GET | Mesh (노드 좌표 + 삼각형 인덱스) |
| `/api/v1/schism/uv?date={yyyyMMdd}&time={HHmm}` | GET | UV (해류 U/V 성분) |

### 공통 응답 헤더

| Header | Value | Description |
|--------|-------|-------------|
| `Content-Type` | `application/octet-stream` | 바이너리 응답 |
| `X-Mesh-Id` | `{nodeCount}-{yyyyMMdd}` | mesh 식별자 (캐시 무효화용) |
| `Cache-Control` | `public, max-age=86400` | mesh만 해당 (1일 캐시) |

클라이언트는 `X-Mesh-Id`를 비교하여 mesh가 변경되었는지 판단한다.
mesh가 동일하면 UV만 갱신하면 된다.

---

## 1. Mesh Binary Format

전체 바이트 순서: **Little-Endian**

```
┌─────────────────────────────────────────────────┐
│ Header (16 bytes)                               │
├─────────────────────────────────────────────────┤
│ Nodes  (nodeCount × 16 bytes)                   │
├─────────────────────────────────────────────────┤
│ Triangles (triangleCount × 12 bytes)            │
└─────────────────────────────────────────────────┘
```

### 1.1 Header (16 bytes)

| Offset | Size | Type   | Field          | Description |
|--------|------|--------|----------------|-------------|
| 0      | 4    | uint32 | nodeCount      | 노드(꼭짓점) 수 |
| 4      | 4    | uint32 | triangleCount  | 삼각형 수 |
| 8      | 8    | -      | reserved       | 예약 (0으로 채움, 향후 확장용) |

### 1.2 Nodes (nodeCount × 16 bytes)

각 노드는 16 bytes:

| Offset | Size | Type    | Field | Description |
|--------|------|---------|-------|-------------|
| 0      | 8    | float64 | lon   | 경도 (EPSG:4326) |
| 8      | 8    | float64 | lat   | 위도 (EPSG:4326) |

```
Node[0]: [lon0, lat0]
Node[1]: [lon1, lat1]
...
Node[N-1]: [lonN-1, latN-1]
```

### 1.3 Triangles (triangleCount × 12 bytes)

각 삼각형은 12 bytes, 노드 인덱스 3개:

| Offset | Size | Type   | Field | Description |
|--------|------|--------|-------|-------------|
| 0      | 4    | uint32 | n0    | 첫 번째 노드 인덱스 (0-based) |
| 4      | 4    | uint32 | n1    | 두 번째 노드 인덱스 (0-based) |
| 8      | 4    | uint32 | n2    | 세 번째 노드 인덱스 (0-based) |

인덱스는 **0-based**. Nodes 배열의 인덱스를 직접 참조한다.

### 1.4 크기 계산

```
총 크기 = 16 + (nodeCount × 16) + (triangleCount × 12)

예시 (실제 SCHISM mesh):
  nodeCount ≈ 266,876
  triangleCount ≈ 496,487
  총 크기 ≈ 16 + 4,270,016 + 5,957,844 ≈ 10.2 MB
```

---

## 2. UV Binary Format

전체 바이트 순서: **Little-Endian**

헤더 없음. 노드 순서대로 U/V 쌍이 나열된다.

```
┌─────────────────────────────────────────────────┐
│ UV Data (nodeCount × 16 bytes)                  │
└─────────────────────────────────────────────────┘
```

### 2.1 UV Data (nodeCount × 16 bytes)

각 노드의 UV는 16 bytes:

| Offset | Size | Type    | Field | Description |
|--------|------|---------|-------|-------------|
| 0      | 8    | float64 | u     | 동서 방향 유속 (m/s, 양수=동쪽) |
| 8      | 8    | float64 | v     | 남북 방향 유속 (m/s, 양수=북쪽) |

```
[u0, v0, u1, v1, u2, v2, ..., uN-1, vN-1]
```

노드 순서는 mesh의 Nodes 배열과 동일하다.
즉 `UV[i]`는 `Node[i]`의 해류 벡터이다.

### 2.2 크기 계산

```
총 크기 = nodeCount × 16

예시: 266,876 노드 × 16 = 4,270,016 ≈ 4.3 MB
```

---

## 3. JavaScript 파싱 예제

```javascript
// ── Mesh 파싱 ──
async function parseMesh(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const nodeCount = view.getUint32(0, true);       // little-endian
  const triangleCount = view.getUint32(4, true);

  const nodesOffset = 16;
  const nodes = new Float64Array(arrayBuffer, nodesOffset, nodeCount * 2);
  // nodes[i*2] = lon, nodes[i*2+1] = lat

  const trianglesOffset = nodesOffset + nodeCount * 16;
  const triangles = new Uint32Array(arrayBuffer, trianglesOffset, triangleCount * 3);
  // triangles[i*3] = n0, triangles[i*3+1] = n1, triangles[i*3+2] = n2

  return { nodeCount, triangleCount, nodes, triangles };
}

// ── UV 파싱 ──
function parseUv(arrayBuffer, nodeCount) {
  const uv = new Float64Array(arrayBuffer, 0, nodeCount * 2);
  // uv[i*2] = u, uv[i*2+1] = v
  return uv;
}
```

### WebAssembly (Rust) 참고

```rust
// Mesh header
let node_count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
let tri_count  = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

// Nodes: &[f64] 로 재해석
let nodes_start = 16;
let nodes_end   = nodes_start + node_count * 16;
let nodes: &[f64] = bytemuck::cast_slice(&data[nodes_start..nodes_end]);

// Triangles: &[u32] 로 재해석
let tris_start = nodes_end;
let tris_end   = tris_start + tri_count * 12;
let triangles: &[u32] = bytemuck::cast_slice(&data[tris_start..tris_end]);

// UV: &[f64] 로 재해석
let uv: &[f64] = bytemuck::cast_slice(&uv_data);
```

---

## 4. 시간 스텝 규칙

| 항목 | 값 |
|------|-----|
| 파일 시작 시각 | 00:10 |
| 스텝 간격 | 10분 |
| 최대 스텝 | 143 (= 23:50) |
| time 파라미터 | HHmm (예: `0010`, `1200`, `2350`) |

계산: `timeStep = (minutes_since_00:10) / 10`, clamped to [0, 143]

---

## 5. 날짜 폴백

요청한 날짜의 파일이 없으면 최대 **2일 전**까지 자동 폴백한다.
실제 사용된 날짜는 `X-Mesh-Id` 헤더의 날짜 부분으로 확인 가능하다.

```
요청: date=20260224
탐색 순서: 20260224 → 20260223 → 20260222
모두 없으면: HTTP 404
```

---

## 6. 좌표계

- 노드 좌표: **EPSG:4326** (WGS84, 경위도)
- 지도 표시 시 EPSG:3857 (Web Mercator)로 변환 필요
