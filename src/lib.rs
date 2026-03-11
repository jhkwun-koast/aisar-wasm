mod coordinate;
mod tiling_system;
mod tile;
mod util;
mod interpolate;
mod mesh;
mod colormap;

use std::cell::RefCell;
use std::cmp::{Ordering, PartialEq};
use std::collections::HashMap;
use std::rc::Rc;
use js_sys::{Uint8Array};
use rstar::{RTree, AABB};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use tiling_system::{ TilingSystem};
use rand::SeedableRng;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use web_sys::{console, js_sys, window, HtmlCanvasElement, CanvasRenderingContext2d};

//use chrono::{DateTime, Utc};

use crate::coordinate::{web_mercator_to_wgs84, wgs84_to_web_mercator, Coordinate, WeatherData};
use crate::interpolate::{interpolate, interpolate_by_bicubic, interpolate_by_inverse_distance_weighted};
use crate::mesh::{MeshNode, TriangleRef, mesh_interpolate_uv, triangle_area};
use crate::tile::Tile;
use crate::util::{calculate_pixel_index, to_coord, to_pixel};

macro_rules! log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()));
}

#[derive(Deserialize)]
pub struct CurrentVectorWrapperOption {
    data: Vec<(f64, f64, f64, f64)>,
    /*mask_data: Vec<u8>,*/
    extent: [f64; 4],
    resolution: f64,
    size: [f64; 2],
}

#[derive(Deserialize)]
pub struct DrawCurrentVectorCanvasOption {
    extent: [f64; 4],
    resolution: f64,
    size: [f64; 2],
    sampling_type: String,
    systematic_value: usize,
    proportional_value: f64,
    dbscan_eps_value: f64,
    dbscan_minpoint_value: usize,
    pixelRatio: f64,
    debug: bool
}

#[derive(Deserialize, Debug)]
pub struct CurrentFlowWrapperOption {
    particle_count:usize,
    data: Vec<(f64, f64, f64, f64)>,
    interpolation_type:String,
    extent: [f64; 4],
    resolution: f64,
    pixel_ratio: f64,
    size: [f64; 2],
    life: i16,
    exaggeration: f64,
    prev_count: i16
}

#[derive(Deserialize)]
pub struct DrawCurrentFlowCanvasOption {
    line_width: f64,
    life: i16,
    exaggeration: f64
}

// 동적 파티클 수 관리 상수
const TARGET_SPACING: f64 = 50.0;
const MIN_PARTICLES: usize = 1000;
const MAX_PARTICLES: usize = 5000;
const TRANSITION_RATE: usize = 100;

// 셀 격자 크기 (픽셀)
const CELL_SIZE: f64 = 50.0;
// 셀당 최대 파티클 수 (밀도 rejection 상한)
const MAX_PARTICLES_PER_CELL: u32 = 5;
// 프레임당 최대 spawn 수 (블로킹 방지)
const MAX_SPAWNS_PER_FRAME: usize = 50;

/// 뷰포트 내 visible 삼각형 캐시 (R-tree 반복 쿼리 제거)
/// viewport 변경 시 1회 빌드, spawn 시 binary search로 O(log N) 선택
struct VisibleTriangleCache {
    triangles: Vec<TriangleRef>,
    cumulative_areas: Vec<f64>,
    total_area: f64,
}

/// Viewport: 화면 좌표 변환 + 셀 격자 관리
#[derive(Debug, Clone)]
struct Viewport {
    canvas_w: f64,
    canvas_h: f64,
    extent: SimpleBounds, // Web Mercator (EPSG:3857)
    grid_cols: usize,
    grid_rows: usize,
}

impl Viewport {
    fn new(canvas_w: f64, canvas_h: f64, extent: SimpleBounds) -> Self {
        let grid_cols = (canvas_w / CELL_SIZE).ceil() as usize;
        let grid_rows = (canvas_h / CELL_SIZE).ceil() as usize;
        Viewport { canvas_w, canvas_h, extent, grid_cols, grid_rows }
    }

    /// Mercator → screen pixel
    fn geo_to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        let sx = (x - self.extent.min_x) / (self.extent.max_x - self.extent.min_x) * self.canvas_w;
        let sy = (self.extent.max_y - y) / (self.extent.max_y - self.extent.min_y) * self.canvas_h;
        (sx, sy)
    }

    /// screen pixel → Mercator
    fn screen_to_geo(&self, sx: f64, sy: f64) -> (f64, f64) {
        let x = self.extent.min_x + sx / self.canvas_w * (self.extent.max_x - self.extent.min_x);
        let y = self.extent.max_y - sy / self.canvas_h * (self.extent.max_y - self.extent.min_y);
        (x, y)
    }

    /// screen pixel → cell (col, row)
    fn screen_to_cell(&self, sx: f64, sy: f64) -> (usize, usize) {
        let col = (sx / CELL_SIZE).floor() as usize;
        let row = (sy / CELL_SIZE).floor() as usize;
        (col.min(self.grid_cols.saturating_sub(1)), row.min(self.grid_rows.saturating_sub(1)))
    }

    /// 화면 크기 기반 목표 파티클 수 산출
    fn compute_target_count(&self) -> usize {
        let count = ((self.canvas_w * self.canvas_h) / (TARGET_SPACING * TARGET_SPACING)) as usize;
        count.clamp(MIN_PARTICLES, MAX_PARTICLES)
    }
}

// 직렬화 가능한 형태로 변환
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct SimpleBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

// AABB에서 간단히 변환된 정보를 생성하는 함수
impl SimpleBounds {
    pub fn to_aabb(&self) -> AABB<WeatherData> {
        let lower_point = WeatherData {
            coordinate: Coordinate::from_array(self.lower()),
            udata: 0.0,
            vdata: 0.0
        };
        let upper_point = WeatherData {
            coordinate: Coordinate::from_array(self.upper()),
            udata: 0.0,
            vdata: 0.0
        };

        AABB::from_corners(lower_point, upper_point)
    }

    pub fn from_aabb_point(aabb: &AABB<WeatherData>) -> Self {
        let lower_point = aabb.lower();
        let upper_point = aabb.upper();
        let lower_coordinate = lower_point.coordinate;
        let upper_coordinate = upper_point.coordinate;
        let lower_lon = lower_coordinate.longitude;
        let lower_lat = lower_coordinate.latitude;
        let upper_lon = upper_coordinate.longitude;
        let upper_lat = upper_coordinate.latitude;

        SimpleBounds { min_x: lower_lon, min_y: lower_lat, max_x: upper_lon, max_y: upper_lat }
    }

    pub fn from_corners(lower: [f64; 2], upper: [f64; 2]) -> Self {
        let [min_x, min_y] = lower;
        let [max_x, max_y] = upper;
        SimpleBounds { min_x, min_y, max_x, max_y }
    }

    pub fn from_extent(extent: [f64;4]) -> Self {
        let [min_x, min_y, max_x, max_y] = extent;
        SimpleBounds { min_x, min_y, max_x, max_y }
    }

    pub fn lower(&self) -> [f64; 2] {
        [self.min_x, self.min_y]
    }

    pub fn upper(&self) -> [f64; 2] {
        [self.max_x, self.max_y]
    }

    // 타일의 너비 계산
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    // 타일의 높이 계산
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn contains(&self, point: &WeatherData) -> bool {
        self.contains_coordinate(point.coordinate)
    }

    pub fn contains_coordinate(&self, coordinate: Coordinate) -> bool {
        let longitude = coordinate.longitude;
        let latitude = coordinate.latitude;

        longitude >= self.min_x && longitude <= self.max_x && latitude >= self.min_y && latitude <= self.max_y
    }

    pub fn contains_bounds(&self, bound: SimpleBounds) -> bool {
        self.min_x <= bound.min_x &&
        self.min_y <= bound.min_y &&
        self.max_x >= bound.max_x &&
        self.max_y >= bound.max_y
    }
}
/*
fn calculate_solar_position(lat: f64, lon: f64, timestamp: f64) -> f64 {
    let pos = sun::pos(timestamp as i64, lat, lon);
    pos.altitude.to_degrees()
}

// WebAssembly로 사용할 함수
#[wasm_bindgen]
pub fn calculateTerminatorCoordinates(timestamp: f64) -> Vec<u8> {
    let start_lat = 23.0;
    let end_lat = 53.0;
    let start_lon = 116.0;
    let end_lon = 143.0;

    let mut terminator_coordinates: Vec<(f64, f64)> = Vec::new();
    log!("timestamp {:?}", timestamp);
    let mut lat = start_lat;
    while lat <= end_lat {
        let mut min_altitude_start = f64::INFINITY;
        let mut min_lon = None;
        let mut min_lat = None;

        let mut lon = start_lon;
        while lon <= end_lon {
            //log!("test lon {:?}, lat {:?}", lon, lat);
            let altitude = calculate_solar_position(lat, lon, timestamp);

            if altitude.abs() < min_altitude_start.abs() {
                min_altitude_start = altitude;
                min_lon = Some(lon);
                min_lat = Some(lat);
            }

            // 경도 0.01 증가
            lon += 0.01;
        }

        if let (Some(min_lon), Some(min_lat)) = (min_lon, min_lat) {
            terminator_coordinates.push((min_lon, min_lat));
        }

        // 위도 0.01 증가
        lat += 0.01;
    }

    let mut serialized: Vec<u8> = Vec::new();

    let count = terminator_coordinates.len() as u32;
    serialized.extend_from_slice(&count.to_le_bytes());

    // 각 (lon, lat) 값을 f64로 직렬화하여 저장
    for (lon, lat) in terminator_coordinates {
        serialized.extend_from_slice(&lon.to_le_bytes());
        serialized.extend_from_slice(&lat.to_le_bytes());
    }

    serialized
}*/

#[wasm_bindgen]
pub struct CurrentVectorWrapper {
    //tiling_system: Rc<RefCell<TilingSystem>>,  // Rc<RefCell<TilingSystem>>로 감쌈
    rtree: Rc<RefCell<RTree<WeatherData>>>,
    bounds: Rc<RefCell<SimpleBounds>>,
    rng: Xoshiro256PlusPlus,
}
fn log_with_time() -> f64 {
    let window = window().expect("no global `window` exists");
    let performance = window.performance().expect("performance should be available");
    performance.now() // 현재 시간을 밀리초 단위로 가져옴
}
fn binary_to_current_vector_wrapper_option(binary_data: Uint8Array) -> CurrentVectorWrapperOption {
    let data_vec = binary_data.to_vec();
    let mut offset = 0;

    // 1. 헤더에서 dataArrayLength와 maskDataLength 읽기
    let data_array_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    /*let mask_data_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;*/
    //log!("data_array_length {:?}, mask_data_length {:?}, offset {:?}", data_array_length, mask_data_length, offset);
    let mut data = Vec::new();

    // 2. dataArray 해석 (f64 4개씩: longitude, latitude, udata, vdata)
    for _ in 0..data_array_length {
        let longitude = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let latitude = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let udata = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let vdata = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;
        //log!("longitude {:?}, latitude {:?}, udata {:?}, vdata  {:?}, offset  {:?}", longitude, latitude, udata, vdata, offset);
        data.push((longitude, latitude, udata, vdata));
    }

    // 3. mask_data 해석
    /*let mask_data = data_vec[offset..offset + mask_data_length].to_vec();
    offset += mask_data_length;*/
    //log!("mask_data length {:?}, offset  {:?}", mask_data.len(), offset);

    // 4. extent 해석 (4개의 64비트 float)
    let mut extent = [0.0; 4];
    for i in 0..4 {
        extent[i] = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;
    }

    //log!("extent {:?}, offset  {:?}", extent, offset);

    // 5. resolution 해석 (64비트 float)
    let resolution = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
    offset += 8;

    //log!("resolution {:?}, offset  {:?}", resolution, offset);

    // 6. size 해석 (64비트 float 2개)
    let mut size = [0.0; 2];
    for i in 0..2 {
        size[i] = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;
    }

    //log!("size {:?}, offset  {:?}", size, offset);

    CurrentVectorWrapperOption {
        data,
        /*mask_data,*/
        extent,
        resolution,
        size,
    }
}
#[wasm_bindgen]
impl CurrentVectorWrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(binary_data: Uint8Array) -> CurrentVectorWrapper {
        let opts: CurrentVectorWrapperOption = binary_to_current_vector_wrapper_option(binary_data);

        let data = opts.data;
        /*let mask_data = opts.mask_data;*/
        let extent = opts.extent;
        let resolution = opts.resolution;
        let size = opts.size;
        let (rtree, bounds) = initialize_data(data, SimpleBounds::from_extent(extent), resolution);
        //log!("rtree {:?}, bounds {:?}", rtree, bounds);
        CurrentVectorWrapper {
            rtree: Rc::new(RefCell::new(rtree)),
            bounds: Rc::new(RefCell::new(bounds)),
            rng: Xoshiro256PlusPlus::from_rng(rand::rngs::OsRng).unwrap(),
        }
    }

    fn points_within_eps(&self, rtree_ref:&RTree<WeatherData>, point: WeatherData, eps: f64) -> Vec<WeatherData> {
        let coord = point.coordinate;
        let x = coord.longitude;
        let y = coord.latitude;
        let sb = SimpleBounds {
            min_x: x - eps,
            min_y: y - eps,
            max_x: x + eps,
            max_y: y + eps,
        };

        // R-tree에서 eps 거리 내 이웃을 찾음
        /*rtree_ref
            .locate_within_distance(point, eps)*/
        rtree_ref.locate_in_envelope(&sb.to_aabb())
            .cloned()
            .collect()
    }
    fn dbscan(&self, eps: f64, min_points: usize) -> Vec<Vec<WeatherData>> {
        let rtree_ref = self.rtree.borrow();
        let points: Vec<WeatherData> = rtree_ref.iter().cloned().collect();

        let mut point_index_map: HashMap<usize, WeatherData> = points.into_iter().enumerate().collect();

        let mut clusters: Vec<Vec<WeatherData>> = Vec::new();
        let mut cluster_id = 0;
        let eps_speed = 0.2;
        let eps_direction = 0.4;

        while let Some((i, point)) = point_index_map.iter().next().map(|(&i, point)| (i, point.clone())) {
            // 거리 기반 이웃 찾기
            let neighbors = self.points_within_eps(&*rtree_ref, point.clone(), eps);

            // 속도 및 방향 기준을 적용한 유효한 이웃 필터링
            /*let valid_neighbors: Vec<_> = neighbors.into_iter()
                .filter(|neighbor| {
                    let speed_diff = (point.speed() - neighbor.speed()).abs();
                    let direction_diff = point.direction_difference(neighbor);
                    speed_diff <= eps_speed && direction_diff <= eps_direction
                })
                .collect();*/

            if neighbors.len() < min_points {
                // 노이즈 포인트 처리
                clusters.push(vec![point.clone()]);
                point_index_map.remove(&i); // 처리된 포인트는 HashMap에서 제거
                continue;
            }

            // 새로운 클러스터 생성
            clusters.push(Vec::new());
            clusters[cluster_id].push(point.clone());

            // 이웃들과 클러스터링 처리
            for neighbor in neighbors {
                let idx = point_index_map.iter().find_map(|(idx, p)| if *p == neighbor { Some(*idx) } else { None });
                if let Some(idx) = idx {
                    clusters[cluster_id].push(point_index_map.remove(&idx).unwrap());
                }
            }

            // 현재 포인트도 클러스터링 완료 후 제거
            point_index_map.remove(&i);
            cluster_id += 1;
        }

        clusters
    }

    pub fn drawCanvasCurrentVecor(&mut self, options:JsValue) -> Result<HtmlCanvasElement, JsValue> {
        let opts: DrawCurrentVectorCanvasOption = serde_wasm_bindgen::from_value(options)?;
        let size = opts.size;
        let raw_extent = opts.extent;
        let resolution = opts.resolution;
        let sampling_type = opts.sampling_type.as_str();
        let systematic_value = opts.systematic_value;
        let proportional_value = opts.proportional_value;
        let dbscan_eps_value = opts.dbscan_eps_value;
        let dbscan_minpoint_value = opts.dbscan_minpoint_value;
        let pixelRatio = opts.pixelRatio;
        let debug = opts.debug;

        let extent = SimpleBounds {
            min_x: raw_extent[0],
            min_y: raw_extent[1],
            max_x: raw_extent[2],
            max_y: raw_extent[3],
        };

        let window = window().ok_or("Could not obtain window")?;
        let document = window.document().ok_or("Could not obtain document")?;

        let canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;
        let width = size[0] as u32;
        let height = size[1] as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let ctx = canvas.get_context("2d")?.unwrap().dyn_into::<CanvasRenderingContext2d>()?;
        ctx.scale(pixelRatio, pixelRatio);
        ctx.clear_rect(0.0, 0.0, width as f64, height as f64);
        let rtree_borrowed = self.rtree.borrow();

        if sampling_type == "dbscan" {
            let mut a = 0;
            let mut b = 0;
            let sample_size = 3;
            let cluster_points_vector = &self.dbscan(dbscan_eps_value, dbscan_minpoint_value);

            for wd_group in cluster_points_vector {
                if wd_group.len() > 1 {
                    b += 1;
                    log!("cluster size {:?}", wd_group.len());
                }
                let sample_size = sample_size.min(wd_group.len()); // 클러스터의 포인트보다 큰 샘플은 방지
                let random_indices = get_random_index(wd_group.len(), sample_size, &mut self.rng); // 랜덤 인덱스 생성

                let sample: Vec<WeatherData> = random_indices
                    .iter()
                    .map(|&i| wd_group[i].clone()) // 인덱스를 기반으로 포인트 복사
                    .collect();

                for wd in sample {
                    draw_vector(&ctx, wd, extent, resolution, debug);
                    a += 1;
                }
            }
            log!("render point len {:?}, cluster len {:?}", a, b);
        } else {
            for data in rtree_borrowed.iter()
                .enumerate()
                .filter_map(|(index, weather_data)| {
                    if sampling_type == "systematic" {
                        if index % systematic_value == 0 {
                            Some(weather_data)  // 선택된 데이터 반환
                        } else {
                            None  // 선택되지 않은 데이터는 제외
                        }
                    } else if sampling_type == "proportional" {
                        let total_points = rtree_borrowed.size(); // 총 포인트 개수
                        let ratio = proportional_value as usize;

                        if ratio == 100 {
                            return Some(weather_data);
                        }

                        // 균등한 간격으로 포인트를 선택할 간격 계산
                        let points_to_take = total_points * ratio / 100; // 선택할 포인트 개수
                        let step = total_points / points_to_take.max(1); // 간격 계산 (0 나누기 방지)

                        if index % step == 0 {
                            return Some(weather_data);
                        }
                        return None;
                    } else {
                        None
                    }
                }) {
                draw_vector(&ctx, *data, extent, resolution, debug);
            }
        }

        Ok(canvas)
    }
}
#[derive(Clone, Debug)]
enum ParticleStatus {
    OK,
    RESPAWN,
    NO
}

impl PartialEq for ParticleStatus {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ParticleStatus::OK, ParticleStatus::OK) => true,
            (ParticleStatus::RESPAWN, ParticleStatus::RESPAWN) => true,
            (ParticleStatus::NO, ParticleStatus::NO) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct Particle {
    coordinate: Coordinate,
    original_coordinate: Coordinate,
    prev_coordinats: Vec<Coordinate>,
    current_u: f64,
    current_v: f64,
    loop_count: i16,
    avoid_interpolation_frame_count: Option<i16>,
    status: ParticleStatus,
    life: i16,
    prev_coordinate: Coordinate, // Phase 2: 이전 프레임 위치 (screen format용)
}

impl Particle {
    pub fn get_speed(&self) -> f64 {
        (self.current_u.powi(2) + self.current_v.powi(2)).sqrt()
    }

    fn f64_to_bytes(value: f64) -> [u8; 8] {
        value.to_le_bytes()
    }
    fn i16_to_bytes(value: i16) -> [u8; 2] {
        value.to_le_bytes()
    }
    fn enum_to_bytes(status: &ParticleStatus) -> [u8; 1] {
        match status {
            ParticleStatus::OK => [0],
            ParticleStatus::RESPAWN => [1],
            ParticleStatus::NO => [2],
        }
    }
    // Coordinate를 직렬화하는 함수
    fn serialize_coordinate(coordinate: &Coordinate) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16); // f64는 8바이트, 총 16바이트 필요
        bytes.extend_from_slice(&Self::f64_to_bytes(coordinate.longitude));
        bytes.extend_from_slice(&Self::f64_to_bytes(coordinate.latitude));
        bytes
    }

    /// Phase 2: 40바이트 고정 포맷 직렬화 (viewport 설정 시 사용)
    /// [prev_x: f64][prev_y: f64][curr_x: f64][curr_y: f64][speed: f32][life_ratio: f32]
    pub fn serialize_screen_format(&self, viewport: &Viewport) -> [u8; 40] {
        let mut buf = [0u8; 40];
        let (prev_sx, prev_sy) = viewport.geo_to_screen(self.prev_coordinate.longitude, self.prev_coordinate.latitude);
        let (curr_sx, curr_sy) = viewport.geo_to_screen(self.coordinate.longitude, self.coordinate.latitude);
        let speed = self.get_speed() as f32;
        let life_ratio = if self.life > 0 {
            (self.life - self.loop_count).max(0) as f32 / self.life as f32
        } else {
            0.0
        };

        buf[0..8].copy_from_slice(&prev_sx.to_le_bytes());
        buf[8..16].copy_from_slice(&prev_sy.to_le_bytes());
        buf[16..24].copy_from_slice(&curr_sx.to_le_bytes());
        buf[24..32].copy_from_slice(&curr_sy.to_le_bytes());
        buf[32..36].copy_from_slice(&speed.to_le_bytes());
        buf[36..40].copy_from_slice(&life_ratio.to_le_bytes());
        buf
    }

    // Particle의 중요한 필드를 직렬화하는 함수
    pub fn serialize_to_vec_u8(&self) -> Vec<u8> {
        let mut serialized_data = Vec::new();

        // coordinate 직렬화
        serialized_data.extend_from_slice(&Self::serialize_coordinate(&self.coordinate));

        serialized_data.extend_from_slice(&Self::f64_to_bytes(self.prev_coordinats.len() as f64));
        // prev_coordinats 직렬화
        for coord in &self.prev_coordinats {
            serialized_data.extend_from_slice(&Self::serialize_coordinate(coord));
        }

        // current_u와 current_v 직렬화
        serialized_data.extend_from_slice(&Self::f64_to_bytes(self.current_u));
        serialized_data.extend_from_slice(&Self::f64_to_bytes(self.current_v));
        serialized_data.extend_from_slice(&Self::enum_to_bytes(&self.status));
        serialized_data.extend_from_slice(&Self::i16_to_bytes(self.life));
        serialized_data.extend_from_slice(&Self::i16_to_bytes(self.loop_count));
        serialized_data
    }
}

fn binary_to_current_flow_wrapper_option(binary_data: Uint8Array) -> CurrentFlowWrapperOption {
    let data_vec = binary_data.to_vec();
    let mut offset = 0;

    // 1. dataArrayLength와 maskDataLength 읽기
    let data_array_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let mut data = Vec::new();

    // 2. dataArray 해석
    for _ in 0..data_array_length {
        let longitude = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let latitude = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let udata = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let vdata = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;

        data.push((longitude, latitude, udata, vdata));
    }

    // 4. extent 해석
    let mut extent = [0.0; 4];
    for i in 0..4 {
        extent[i] = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;
    }

    // 5. resolution 해석
    let resolution = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // 6. size 해석
    let mut size = [0.0; 2];
    for i in 0..2 {
        size[i] = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
        offset += 8;
    }

    // 7. particle_count 해석 (8바이트)
    let particle_count = u64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let pixel_ratio = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // 9. interpolation_type 해석 (문자열)
    let string_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let string_bytes = &data_vec[offset..offset + string_length];
    let interpolation_type = String::from_utf8(string_bytes.to_vec()).unwrap();
    offset += string_length;

    let life = i16::from_le_bytes(data_vec[offset..offset + 2].try_into().unwrap());
    offset += 2;

    let prev_count = i16::from_le_bytes(data_vec[offset..offset + 2].try_into().unwrap());
    offset += 2;

    let exaggeration = f64::from_le_bytes(data_vec[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // CurrentVectorWrapperOption 구조체 반환
    CurrentFlowWrapperOption {
        data,
        extent,
        resolution,
        size,
        particle_count,
        pixel_ratio,
        interpolation_type,
        life,
        exaggeration,
        prev_count
    }
}

#[wasm_bindgen]
pub struct CurrentFlowWrapper {
    tiling_system: Rc<RefCell<TilingSystem>>,  // Rc<RefCell<TilingSystem>>로 감쌈
    particles: Vec<Particle>,
    particle_count: usize,
    rendering_count: usize,
    interpolation_type: String,
    extent: [f64; 4],
    resolution: f64,
    pixel_ratio: f64,
    size: [f64; 2],
    depth: usize,
    life: i16,
    exaggeration: f64,
    prev_count: i16,

    // 삼각형 메시 필드 (Phase 1)
    mesh_nodes: Vec<MeshNode>,
    tri_rtree: Option<RTree<TriangleRef>>,
    mesh_bounds: Option<SimpleBounds>,

    // Phase 2: Viewport
    viewport: Option<Viewport>,
    // Phase 2: 메시 커버리지 셀 마스크 (뷰포트+메시 기반)
    mesh_cell_mask: Option<Vec<bool>>,
    // visible 삼각형 캐시 (뷰포트 변경 시 무효화, R-tree 반복 쿼리 제거)
    visible_tri_cache: Option<VisibleTriangleCache>,

    // 산술 PRNG (crypto.getRandomValues 대체)
    rng: Xoshiro256PlusPlus,
}

#[wasm_bindgen]
impl CurrentFlowWrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(binary_data:Uint8Array ) -> CurrentFlowWrapper {
        log!("[WASM] CurrentFlowWrapper::new v2 (hang fix applied)");
        // OsRng로 시드 1회만 생성 (crypto.getRandomValues 1회 호출)
        let mut rng = Xoshiro256PlusPlus::from_rng(rand::rngs::OsRng).unwrap();
        let opts: CurrentFlowWrapperOption = binary_to_current_flow_wrapper_option(binary_data);

        let particle_count = opts.particle_count;
        let mut interpolation_type = opts.interpolation_type;
        let extent = opts.extent;
        let resolution = opts.resolution;
        let size = opts.size;
        let pixel_ratio = opts.pixel_ratio;
        let data = opts.data;
        let life = opts.life;
        let exaggeration = opts.exaggeration;
        let prev_count = opts.prev_count;

        // data가 비어 있거나, size/resolution이 0이면 파티클 생성을 건너뜀
        // (빈 R-tree에서 역전된 bounds로 무한루프 방지)
        // load_mesh_binary() 호출 시 파티클이 재생성됨
        let can_generate = !data.is_empty() && size[0] > 0.0 && size[1] > 0.0 && resolution > 0.0;
        let pc = if can_generate { Some(particle_count) } else { None };
        log!("[WASM] data.len={}, size={:?}, resolution={}, particle_gen={}", data.len(), size, resolution, can_generate);
        let (rtree, bounds, particles) = initialize_data_with_particle(data, SimpleBounds::from_extent(extent), resolution, size, pc, life, &mut rng);

        let envelope = rtree.root().envelope();
        let upper = envelope.upper();
        let lower = envelope.lower();

        let width = (upper.coordinate.longitude - lower.coordinate.longitude) / 40.0;
        log!("width {:?}", width);

        if rtree.size() < 16 {
            interpolation_type = "NEAREST".parse().unwrap();
        }
        let first_tile = Tile {rtree, bounds, depth: 0 };

        // 타일링 시스템 초기화
        let mut tiling_system = TilingSystem::new(0);

        tiling_system.tiles_by_depth.entry(0)
            .or_insert_with(Vec::new)
            .push(first_tile);

        CurrentFlowWrapper {
            tiling_system: Rc::new(RefCell::new(tiling_system)),
            particles: particles.unwrap_or(Vec::new()),
            particle_count,
            rendering_count: 0,
            interpolation_type,
            extent,
            resolution,
            pixel_ratio,
            size,
            depth: 0,
            life,
            exaggeration,
            prev_count,
            mesh_nodes: Vec::new(),
            tri_rtree: None,
            mesh_bounds: None,
            viewport: None,
            mesh_cell_mask: None,
            visible_tri_cache: None,
            rng,
        }
    }

    pub fn setInterpolationType(&mut self, interpolation_type:String) {
        self.interpolation_type = interpolation_type;
        self.rendering_count = 0;
        for particle in self.particles.iter_mut() {
            particle.prev_coordinats.clear();
            particle.coordinate = particle.original_coordinate;
            particle.current_u = 0.0;
            particle.current_v = 0.0;
            particle.loop_count = 0;
            particle.avoid_interpolation_frame_count = None;
            particle.status = ParticleStatus::OK;
        }
    }

    pub fn adjustParticleCount(&mut self, new_particle_count: usize) {
        let current_count = self.particle_count;

        if current_count < new_particle_count {
            // 파티클 추가
            let particles_to_add = new_particle_count - current_count;
            self.add_particles(particles_to_add);
        } else if current_count > new_particle_count {
            // 파티클 제거
            let particles_to_remove = current_count - new_particle_count;
            self.remove_particles(particles_to_remove);
        }

        self.particle_count = new_particle_count;
    }

    pub fn set_exaggeration(&mut self, value: f64) {
        self.exaggeration = value;
    }

    pub fn reset_particles(&mut self) {
        // 모든 파티클 life = 0 → 다음 프레임에 전부 리스폰
        for p in &mut self.particles {
            p.life = 0;
        }
    }

    /// Viewport 설정: canvas 크기 + extent (Web Mercator EPSG:3857)
    /// extent: Float64Array [min_x, min_y, max_x, max_y]
    pub fn set_viewport(&mut self, canvas_w: f64, canvas_h: f64, extent: &js_sys::Float64Array) {
        let ext = extent.to_vec();
        if ext.len() < 4 {
            return;
        }
        let bounds = SimpleBounds {
            min_x: ext[0],
            min_y: ext[1],
            max_x: ext[2],
            max_y: ext[3],
        };
        self.viewport = Some(Viewport::new(canvas_w, canvas_h, bounds));
        self.mesh_cell_mask = None; // 뷰포트 변경 시 마스크 무효화
        self.visible_tri_cache = None; // 뷰포트 변경 시 삼각형 캐시 무효화
    }

    /// 메시 셀 마스크가 없으면 빌드 (뷰포트+메시 모두 존재할 때)
    fn ensure_mesh_cell_mask(&mut self) {
        if self.mesh_cell_mask.is_some() { return; }
        if let (Some(ref vp), Some(ref rtree)) = (&self.viewport, &self.tri_rtree) {
            let mask = build_mesh_cell_mask(vp, rtree, &self.mesh_nodes);
            let valid = mask.iter().filter(|&&v| v).count();
            log!("[WASM] mesh_cell_mask built: {}x{} grid, {} valid cells out of {}",
                vp.grid_cols, vp.grid_rows, valid, mask.len());
            self.mesh_cell_mask = Some(mask);
        }
    }

    /// visible 삼각형 캐시가 없으면 빌드 (뷰포트+메시 모두 존재할 때)
    /// viewport 내 삼각형 목록 + 누적 면적 배열을 캐시하여 spawn 시 R-tree 재쿼리 제거
    fn ensure_visible_tri_cache(&mut self) {
        if self.visible_tri_cache.is_some() { return; }
        if let (Some(ref vp), Some(ref rtree)) = (&self.viewport, &self.tri_rtree) {
            let nodes = &self.mesh_nodes;
            let (gx_min, gy_min) = vp.screen_to_geo(0.0, vp.canvas_h);
            let (gx_max, gy_max) = vp.screen_to_geo(vp.canvas_w, 0.0);
            let envelope = AABB::from_corners([gx_min, gy_min], [gx_max, gy_max]);

            let mut triangles = Vec::new();
            let mut cumulative_areas = Vec::new();
            let mut total_area = 0.0;

            for tri in rtree.locate_in_envelope_intersecting(&envelope) {
                let area = triangle_area(tri, nodes);
                if area < 1e-20 { continue; }
                total_area += area;
                triangles.push(tri.clone());
                cumulative_areas.push(total_area);
            }

            log!("[WASM] visible_tri_cache built: {} triangles, total_area={:.6}", triangles.len(), total_area);
            self.visible_tri_cache = Some(VisibleTriangleCache {
                triangles,
                cumulative_areas,
                total_area,
            });
        }
    }

    // 파티클을 추가하는 메서드
    fn add_particles(&mut self, num_to_add: usize) {
        let borrow_ref = self.tiling_system.borrow();
        if let Some(root_tiles) = borrow_ref.tiles_by_depth.get(&0) {
            let &unwrap_root_tile = &root_tiles.get(0).unwrap();
            let rtree = &unwrap_root_tile.rtree;
            let rtree_ref = rtree;
            let bounds = rtree_ref.root().envelope();
            let simple_bounds = SimpleBounds::from_aabb_point(&bounds);

            let particles = generate_particles(SimpleBounds::from_extent(self.extent), num_to_add/*, &self.mask_data*/, simple_bounds, self.resolution, self.size, self.life, &mut self.rng);
            self.particles.extend(particles);
        }
    }

    // 파티클을 제거하는 메서드
    fn remove_particles(&mut self, num_to_remove: usize) {
        let mut removed_count = 0;

        self.particles.retain(|_| {
            if removed_count < num_to_remove {
                removed_count += 1; // 타일에서 파티클 제거 카운트 증가
                return false; // 파티클 삭제
            }
            true // 파티클 유지
        });
    }

    /// Phase 2: 프레임마다 호출 — 메시 커버리지 기반으로 파티클 수 점진 조정
    fn adjust_particle_count_gradual(&mut self) {
        let viewport = match &self.viewport {
            Some(v) => v.clone(),
            None => return,
        };

        // 메시 셀 마스크 기반 목표 파티클 수: 메시가 있는 셀만 고려
        let valid_cells = self.mesh_cell_mask.as_ref()
            .map(|m| m.iter().filter(|&&v| v).count())
            .unwrap_or(viewport.grid_cols * viewport.grid_rows);

        let target = if valid_cells == 0 {
            0
        } else {
            let particles_per_cell = (CELL_SIZE / TARGET_SPACING).powi(2) as usize; // = 4
            (valid_cells * particles_per_cell).min(MAX_PARTICLES)
        };

        let current = self.particles.len();

        if current < target {
            let to_add = (target - current).min(TRANSITION_RATE);
            if let (Some(_rtree), Some(_mb), Some(ref cache)) = (&self.tri_rtree, &self.mesh_bounds, &self.visible_tri_cache) {
                let life = self.life;
                let nodes = &self.mesh_nodes;
                let mut cell_counts = build_cell_counts(&self.particles, &viewport);
                let rng = &mut self.rng;
                for _ in 0..to_add {
                    if let Some(p) = spawn_balanced_particle(&viewport, cache, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL, rng) {
                        self.particles.push(p);
                    }
                }
            }
        } else if current > target {
            let to_remove = (current - target).min(TRANSITION_RATE);
            remove_shortest_lived(&mut self.particles, to_remove);
        }
    }

    /// Phase 2: 뷰포트 밖 파티클을 뷰포트 안으로 점진 이동 (매 프레임 TRANSITION_RATE개)
    fn migrate_particles_to_viewport(&mut self) {
        let viewport = match &self.viewport {
            Some(v) => v.clone(),
            None => return,
        };
        let cache = match &self.visible_tri_cache {
            Some(c) => c,
            None => return,
        };
        let life = self.life;
        let nodes = &self.mesh_nodes;

        // 뷰포트 밖 파티클 인덱스 수집
        let mut out_indices: Vec<usize> = Vec::new();
        for (i, p) in self.particles.iter().enumerate() {
            let (sx, sy) = viewport.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
            if sx < 0.0 || sx >= viewport.canvas_w || sy < 0.0 || sy >= viewport.canvas_h {
                out_indices.push(i);
            }
        }

        if out_indices.is_empty() {
            return;
        }

        let to_migrate = out_indices.len().min(TRANSITION_RATE);
        let mut cell_counts = build_cell_counts(&self.particles, &viewport);
        let rng = &mut self.rng;

        for &idx in out_indices.iter().take(to_migrate) {
            if let Some(new_p) = spawn_balanced_particle(&viewport, cache, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL, rng) {
                self.particles[idx] = new_p;
            }
            // spawn 실패 시 그대로 둠 (다음 프레임에 재시도)
        }
    }

    /// 뷰포트 밖 파티클이 50% 이상이면 프레임당 MAX_SPAWNS_PER_FRAME개씩 점진 교체
    /// (블로킹 방지: 이전에는 수천 개를 한 번에 spawn하여 ~105초 블로킹 발생)
    fn bulk_migrate_if_needed(&mut self) {
        // 조건 확인 (borrow scope 분리)
        let should_migrate = {
            let vp = match &self.viewport { Some(v) => v, None => return };
            let total = self.particles.len();
            if total == 0 { return; }
            let out_count = self.particles.iter().filter(|p| {
                let (sx, sy) = vp.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
                sx < 0.0 || sx >= vp.canvas_w || sy < 0.0 || sy >= vp.canvas_h
            }).count();
            out_count * 2 > total
        };

        if !should_migrate { return; }

        // 뷰포트 밖 파티클 인덱스 수집
        let vp = self.viewport.as_ref().unwrap().clone();
        let mut out_indices: Vec<usize> = Vec::new();
        for (i, p) in self.particles.iter().enumerate() {
            let (sx, sy) = vp.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
            if sx < 0.0 || sx >= vp.canvas_w || sy < 0.0 || sy >= vp.canvas_h {
                out_indices.push(i);
            }
        }

        // 프레임당 최대 MAX_SPAWNS_PER_FRAME개만 교체 (나머지는 다음 프레임)
        let to_migrate = out_indices.len().min(MAX_SPAWNS_PER_FRAME);
        if to_migrate == 0 { return; }

        let cache = match &self.visible_tri_cache { Some(c) => c, None => return };
        let life = self.life;
        let nodes = &self.mesh_nodes;
        let mut cell_counts = build_cell_counts(&self.particles, &vp);
        let rng = &mut self.rng;

        for &idx in out_indices.iter().take(to_migrate) {
            if let Some(p) = spawn_balanced_particle(&vp, cache, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL, rng) {
                self.particles[idx] = p;
            }
            // spawn 실패 시 그대로 둠 (다음 프레임에 재시도)
        }
        log!("[WASM] bulk migration: replaced {} of {} out-of-viewport particles", to_migrate, out_indices.len());
    }

    pub fn getUpdateParticles(&mut self) -> Uint8Array {
        if self.tri_rtree.is_some() {
            self.update_particles_mesh()
        } else {
            self.update_particles_legacy()
        }
    }

    fn update_particles_legacy(&mut self) -> Uint8Array {
        let size = self.size;
        let raw_extent = self.extent;
        let extent = SimpleBounds::from_extent(raw_extent);
        let resolution = self.resolution;
        let life = self.life;
        let prev_count = self.prev_count;
        let exaggeration = self.exaggeration;

        let borrow_ref = self.tiling_system.borrow();
        let mut particle_u8s = Vec::new();
        if let Some(root_tiles) = borrow_ref.tiles_by_depth.get(&0) {
            let &unwrap_root_tile = &root_tiles.get(0).unwrap();
            let rtree = &unwrap_root_tile.rtree;

            if rtree.size() < 4 {
                ()
            }
            let rtree_ref = rtree;
            let interp_type = &self.interpolation_type;
            let rng = &mut self.rng;

            for (_, particle) in self.particles.iter_mut().enumerate() {
                let pixel = to_pixel(particle.coordinate.longitude, particle.coordinate.latitude, extent, resolution);
                let _success = handle_particle(particle, rtree_ref, interp_type, &exaggeration, pixel, extent, resolution, size, life, prev_count, rng);
                particle_u8s.push(particle.serialize_to_vec_u8());
            }
        }

        let total_size: usize = particle_u8s.iter().map(|v| v.len()).sum();
        let mut combined_data = Vec::with_capacity(total_size);
        for vec in &particle_u8s {
            combined_data.extend_from_slice(vec);
        }

        Uint8Array::from(&combined_data[..])
    }

    fn update_particles_mesh(&mut self) -> Uint8Array {
        // 메시 셀 마스크 빌드 (뷰포트+메시 존재 시)
        self.ensure_mesh_cell_mask();

        // visible 삼각형 캐시 빌드 (뷰포트 변경 시 1회)
        self.ensure_visible_tri_cache();

        // 뷰포트 밖 파티클이 50% 이상이면 점진 재배치 (프레임당 최대 50개)
        self.bulk_migrate_if_needed();

        // Step 5: 점진 증감 (메시 커버리지 기반)
        self.adjust_particle_count_gradual();

        // Phase 2: 뷰포트 밖 파티클을 뷰포트 안으로 점진 이동
        self.migrate_particles_to_viewport();

        let life = self.life;
        let prev_count = self.prev_count;
        let exaggeration = self.exaggeration;

        let rtree = self.tri_rtree.as_ref().unwrap();
        let nodes = &self.mesh_nodes;
        let mesh_bounds = self.mesh_bounds.unwrap();
        let has_viewport = self.viewport.is_some();
        let frame_number = self.rendering_count;
        self.rendering_count += 1;

        // Step 6: 셀 카운트 빌드 (viewport 있을 때만)
        let mut cell_counts = if let Some(ref vp) = self.viewport {
            build_cell_counts(&self.particles, vp)
        } else {
            Vec::new()
        };

        let rng = &mut self.rng;

        // viewport 있으면 40B × N 고정 크기, 없으면 가변
        if has_viewport {
            let vp = self.viewport.as_ref().unwrap();
            let cache = self.visible_tri_cache.as_ref().unwrap();
            let mut output = Vec::with_capacity(self.particles.len() * 40);

            // 첫 프레임 진단 로그
            if frame_number == 0 {
                let mut none_count = 0u32;
                let mut zero_count = 0u32;
                let mut valid_count = 0u32;
                let mut sample_uv = (0.0f64, 0.0f64);
                let mut in_view = 0u32;
                for p in self.particles.iter() {
                    let (sx, sy) = vp.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
                    if sx >= 0.0 && sx < vp.canvas_w && sy >= 0.0 && sy < vp.canvas_h {
                        in_view += 1;
                    }
                    match mesh_interpolate_uv(p.coordinate.longitude, p.coordinate.latitude, rtree, nodes) {
                        Some((u, v)) => {
                            if u.abs() < 1e-10 && v.abs() < 1e-10 {
                                zero_count += 1;
                            } else {
                                valid_count += 1;
                                if valid_count == 1 {
                                    sample_uv = (u, v);
                                }
                            }
                        }
                        None => { none_count += 1; }
                    }
                }
                log!("[WASM] frame#0 diag: total={}, inView={}, none={}, zero_uv={}, valid_uv={}, sample=({:.6},{:.6}), life={}, exagg={:.4}",
                    self.particles.len(), in_view, none_count, zero_count, valid_count, sample_uv.0, sample_uv.1, life, exaggeration);
            }

            for particle in self.particles.iter_mut() {
                // Deferred respawn: 이전 프레임에서 RESPAWN 마킹된 파티클 처리
                if particle.status == ParticleStatus::RESPAWN {
                    if let Some(new_p) = spawn_balanced_particle(vp, cache, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL, rng) {
                        *particle = new_p;
                    }
                    // None이면 그대로 둠 (극히 드문 경우, 다음 프레임에 재시도)
                }

                // 이전 좌표 저장
                particle.prev_coordinate = particle.coordinate;

                let x = particle.coordinate.longitude;
                let y = particle.coordinate.latitude;

                match mesh_interpolate_uv(x, y, rtree, nodes) {
                    Some((u, v)) => {
                        const EPSILON: f64 = 1e-10;
                        if u.abs() < EPSILON && v.abs() < EPSILON {
                            // 유속이 0에 가까우면 → 삼각형 기반 균등 리스폰
                            if let Some(new_p) = spawn_balanced_particle(vp, cache, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL, rng) {
                                *particle = new_p;
                            }
                        } else {
                            // Mercator 보정: Web Mercator 등각도법 — X, Y 모두 1/cos(φ) 적용
                            let lat_rad = 2.0 * (particle.coordinate.latitude * std::f64::consts::PI / 20037508.34).exp().atan() - std::f64::consts::FRAC_PI_2;
                            let cos_lat = lat_rad.cos();
                            particle.coordinate.longitude += u * exaggeration / cos_lat;
                            particle.coordinate.latitude += v * exaggeration / cos_lat;
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
                                // Deferred respawn: 이동은 보여주고, 다음 프레임에 리스폰
                                particle.status = ParticleStatus::RESPAWN;
                            }
                        }
                    }
                    None => {
                        // 삼각형 밖 → 삼각형 기반 균등 리스폰
                        if let Some(new_p) = spawn_balanced_particle(vp, cache, nodes, life, &mut cell_counts, MAX_PARTICLES_PER_CELL, rng) {
                            *particle = new_p;
                        }
                    }
                }

                // 40B 직렬화
                output.extend_from_slice(&particle.serialize_screen_format(vp));
            }

            Uint8Array::from(&output[..])
        } else {
            // 레거시 가변 포맷 (viewport 없음)
            let mut particle_u8s = Vec::new();

            for particle in self.particles.iter_mut() {
                particle.prev_coordinate = particle.coordinate;

                let x = particle.coordinate.longitude;
                let y = particle.coordinate.latitude;

                match mesh_interpolate_uv(x, y, rtree, nodes) {
                    Some((u, v)) => {
                        const EPSILON: f64 = 1e-10;
                        if u.abs() < EPSILON && v.abs() < EPSILON {
                            *particle = generate_particle_mesh(mesh_bounds, rtree, nodes, life, rng);
                        } else {
                            // Mercator 보정: Web Mercator 등각도법 — X, Y 모두 1/cos(φ) 적용
                            let lat_rad = 2.0 * (particle.coordinate.latitude * std::f64::consts::PI / 20037508.34).exp().atan() - std::f64::consts::FRAC_PI_2;
                            let cos_lat = lat_rad.cos();
                            particle.coordinate.longitude += u * exaggeration / cos_lat;
                            particle.coordinate.latitude += v * exaggeration / cos_lat;
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
                                *particle = generate_particle_mesh(mesh_bounds, rtree, nodes, life, rng);
                                particle.status = ParticleStatus::NO;
                            }
                        }
                    }
                    None => {
                        *particle = generate_particle_mesh(mesh_bounds, rtree, nodes, life, rng);
                    }
                }

                particle_u8s.push(particle.serialize_to_vec_u8());
            }

            let total_size: usize = particle_u8s.iter().map(|v| v.len()).sum();
            let mut combined_data = Vec::with_capacity(total_size);
            for vec in &particle_u8s {
                combined_data.extend_from_slice(vec);
            }

            Uint8Array::from(&combined_data[..])
        }
    }

    /// 삼각형 메시 로드: JS에서 노드 좌표 + 삼각형 인덱스를 전달
    /// nodes_buf: [lon0, lat0, lon1, lat1, ...] (WGS84)
    /// triangles_buf: [n0, n1, n2, ...] (삼각형당 3개 인덱스)
    pub fn load_mesh(&mut self, nodes_buf: &js_sys::Float64Array, triangles_buf: &js_sys::Uint32Array) {
        let nodes_vec = nodes_buf.to_vec();
        let tris_vec = triangles_buf.to_vec();
        let node_count = nodes_vec.len() / 2;
        let tri_count = tris_vec.len() / 3;

        self.build_mesh_internal(node_count, tri_count, |i| {
            (nodes_vec[i * 2], nodes_vec[i * 2 + 1])
        }, |i| {
            [tris_vec[i * 3], tris_vec[i * 3 + 1], tris_vec[i * 3 + 2]]
        });
    }

    /// SCHISM 바이너리 포맷에서 직접 메시 로드
    /// 포맷: [nodeCount:u32][triCount:u32][reserved:8bytes][nodes: nodeCount×16bytes][triangles: triCount×12bytes]
    pub fn load_mesh_binary(&mut self, data: Uint8Array) {
        let buf = data.to_vec();
        if buf.len() < 16 {
            log!("[WASM] load_mesh_binary: 데이터 크기 부족 ({}bytes < 16bytes header)", buf.len());
            return;
        }

        let node_count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let tri_count = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        // buf[8..16] reserved

        let nodes_start = 16;
        let nodes_end = nodes_start + node_count * 16;
        let tris_start = nodes_end;
        let tris_end = tris_start + tri_count * 12;

        let expected_size = tris_end;
        if buf.len() < expected_size {
            log!("[WASM] load_mesh_binary: 데이터 크기 불일치 (실제={}bytes, 필요={}bytes, nodes={}, tris={})",
                buf.len(), expected_size, node_count, tri_count);
            return;
        }

        log!("[WASM] load_mesh_binary: nodes={}, triangles={}, size={}bytes", node_count, tri_count, buf.len());

        let nodes_slice = &buf[nodes_start..nodes_end];
        let tris_slice = &buf[tris_start..tris_end];

        self.build_mesh_internal(node_count, tri_count, |i| {
            let off = i * 16;
            let lon = f64::from_le_bytes(nodes_slice[off..off + 8].try_into().unwrap());
            let lat = f64::from_le_bytes(nodes_slice[off + 8..off + 16].try_into().unwrap());
            (lon, lat)
        }, |i| {
            let off = i * 12;
            let n0 = u32::from_le_bytes(tris_slice[off..off + 4].try_into().unwrap());
            let n1 = u32::from_le_bytes(tris_slice[off + 4..off + 8].try_into().unwrap());
            let n2 = u32::from_le_bytes(tris_slice[off + 8..off + 12].try_into().unwrap());
            [n0, n1, n2]
        });
    }

    /// 공통 메시 구축 로직
    fn build_mesh_internal(
        &mut self,
        node_count: usize,
        tri_count: usize,
        get_node: impl Fn(usize) -> (f64, f64),
        get_tri: impl Fn(usize) -> [u32; 3],
    ) {
        let mut mesh_nodes = Vec::with_capacity(node_count);

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for i in 0..node_count {
            let (lon, lat) = get_node(i);
            let coord = wgs84_to_web_mercator(lon, lat);
            let x = coord.longitude;
            let y = coord.latitude;

            if x < min_x { min_x = x; }
            if y < min_y { min_y = y; }
            if x > max_x { max_x = x; }
            if y > max_y { max_y = y; }

            mesh_nodes.push(MeshNode { x, y, u: 0.0, v: 0.0 });
        }

        let mut triangle_refs = Vec::with_capacity(tri_count);
        for i in 0..tri_count {
            let indices = get_tri(i);
            triangle_refs.push(TriangleRef::new(indices, &mesh_nodes));
        }

        let tri_rtree = RTree::bulk_load(triangle_refs);
        let mesh_bounds = SimpleBounds { min_x, min_y, max_x, max_y };

        self.mesh_nodes = mesh_nodes;
        self.tri_rtree = Some(tri_rtree);
        self.mesh_bounds = Some(mesh_bounds);
        self.mesh_cell_mask = None; // 메시 변경 시 마스크 무효화
        self.visible_tri_cache = None; // 메시 변경 시 삼각형 캐시 무효화

        // 파티클 재생성 (mesh_bounds 기반)
        let rtree = self.tri_rtree.as_ref().unwrap();
        let nodes = &self.mesh_nodes;
        let life = self.life;
        let count = self.particle_count;
        let rng = &mut self.rng;

        let mut particles = Vec::with_capacity(count);
        for _ in 0..count {
            particles.push(generate_particle_mesh(mesh_bounds, rtree, nodes, life, rng));
        }
        self.particles = particles;
    }

    /// UV 데이터 갱신 (Float64Array): [u0, v0, u1, v1, ...] 순서
    pub fn update_uv(&mut self, uv_buf: &js_sys::Float64Array) {
        let uv_vec = uv_buf.to_vec();
        let count = uv_vec.len() / 2;
        if count != self.mesh_nodes.len() {
            log!("[WASM] update_uv: 노드 수 불일치 (UV={}, mesh={}), 최소값 기준 적용", count, self.mesh_nodes.len());
        }
        for i in 0..count.min(self.mesh_nodes.len()) {
            self.mesh_nodes[i].u = uv_vec[i * 2];
            self.mesh_nodes[i].v = uv_vec[i * 2 + 1];
        }
    }

    /// UV 바이너리 데이터 갱신 (Uint8Array): SCHISM UV 포맷 직접 파싱
    /// 포맷: nodeCount × 8bytes [u:f32, v:f32] little-endian, 헤더 없음
    pub fn update_uv_binary(&mut self, data: Uint8Array) {
        let buf = data.to_vec();
        let expected_size = self.mesh_nodes.len() * 8;

        if buf.len() < 8 {
            log!("[WASM] update_uv_binary: 데이터가 비어있음 ({}bytes)", buf.len());
            return;
        }

        let uv_count = buf.len() / 8;
        if buf.len() != expected_size {
            log!("[WASM] update_uv_binary: 크기 불일치 (실제={}bytes/{}nodes, 필요={}bytes/{}nodes)",
                buf.len(), uv_count, expected_size, self.mesh_nodes.len());
        }

        let count = uv_count.min(self.mesh_nodes.len());
        for i in 0..count {
            let off = i * 8;
            let u = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as f64;
            let v = f32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap()) as f64;
            self.mesh_nodes[i].u = u;
            self.mesh_nodes[i].v = v;
        }

        log!("[WASM] update_uv_binary: {}개 노드 UV 갱신 완료", count);
    }

    /// 로드된 메시의 노드 수 반환 (UV 데이터 크기 검증용)
    pub fn get_node_count(&self) -> usize {
        self.mesh_nodes.len()
    }

    /// 메시 로드 여부 확인
    pub fn has_mesh(&self) -> bool {
        self.tri_rtree.is_some()
    }
}

fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

fn handle_particle(
    particle: &mut Particle,
    rtree: &RTree<WeatherData>,
    interpolation_type: &String,
    exaggeration: &f64,
    pixel: [f64; 2],
    /*mask_data: &Vec<u8>,*/
    extent: SimpleBounds,
    resolution: f64,
    size: [f64; 2],
    life: i16,
    prev_count: i16,
    rng: &mut Xoshiro256PlusPlus,
) ->bool {
    let data_bounds = SimpleBounds::from_aabb_point(&rtree.root().envelope());

    // 3. 보간 처리
    let (interpolated_u, interpolated_v) = interpolate_uv(particle, rtree, interpolation_type,  extent, resolution, size);

    const EPSILON: f64 = 1e-10;
    if (approx_equal(interpolated_u, -999.0, EPSILON) && approx_equal(interpolated_v, -999.0, EPSILON)) || ((interpolated_u.abs() < EPSILON) && (interpolated_v.abs() < EPSILON)) {
        let contain_bound = if extent.contains_bounds(data_bounds)  {extent} else {data_bounds};
        *particle = generate_particle(extent/*, mask_data*/, contain_bound, resolution, size, life, rng);
        return false;
    }

    // Mercator 보정: Web Mercator 등각도법 — X, Y 모두 1/cos(φ) 적용
    let lat_rad = 2.0 * (particle.coordinate.latitude * std::f64::consts::PI / 20037508.34).exp().atan() - std::f64::consts::FRAC_PI_2;
    let cos_lat = lat_rad.cos();
    let new_longitude = particle.coordinate.longitude + interpolated_u * exaggeration / cos_lat;
    let new_latitude = particle.coordinate.latitude + interpolated_v * exaggeration / cos_lat;
    let new_pixel = to_pixel(new_longitude, new_latitude, extent, resolution);

    // 정상 수행
    particle.coordinate.longitude = new_longitude;
    particle.coordinate.latitude = new_latitude;
    particle.current_u = interpolated_u;
    particle.current_v = interpolated_v;

    particle.status = ParticleStatus::OK;

    update_particle_state(particle, /*mask_data,*/ data_bounds, extent, resolution,size, life, prev_count, rng);

    true
}
/*
fn handle_particle(
    particle: &mut Particle,
    rtree: &RTree<WeatherData>,
    interpolation_type: &String,
    exaggeration: &f64,
    pixel: [f64; 2],
    /*mask_data: &Vec<u8>,*/
    extent: SimpleBounds,
    resolution: f64,
    size: [f64; 2],
    life: i16
) {
    let width = size[0];
    let height = size[1];

    let data_bounds = SimpleBounds::from_aabb_point(&rtree.root().envelope());

    // 1. 보간을 건너뛰는 경우 처리
    if let Some(avoid_count) = particle.avoid_interpolation_frame_count {
        if avoid_count > 0 {
            particle.coordinate.longitude += particle.current_u * exaggeration;
            particle.coordinate.latitude += particle.current_v * exaggeration;
            particle.avoid_interpolation_frame_count = Some(avoid_count - 1);

            let reflected_pixel = to_pixel(particle.coordinate.longitude, particle.coordinate.latitude, extent, resolution);
            if should_reflect_or_detour(particle, mask_data, reflected_pixel, width as usize, height as usize, exaggeration) {
                update_particle_state(particle, mask_data, data_bounds, extent, resolution, size, life);
                return;
            }
            update_particle_state(particle, mask_data, data_bounds, extent, resolution, size, life);
            return;
        }
    }

    particle.avoid_interpolation_frame_count = None;

    // 2. 반사 또는 우회 여부 확인
    if should_reflect_or_detour(particle, mask_data, pixel, width as usize, height as usize, exaggeration) {
        update_particle_state(particle, mask_data, data_bounds, extent, resolution, size, life);
        return;  // 이미 반사 또는 우회가 이루어졌으면 여기서 종료
    }

    // 3. 보간 처리
    let (interpolated_u, interpolated_v) = interpolate_uv(particle, rtree, interpolation_type, mask_data, extent, resolution, size);

    if interpolated_u == 0.0 && interpolated_v == 0.0 {
        *particle = generate_particle(extent/*, mask_data*/, data_bounds, resolution, size);
        return;
    }

    // Mercator 보정: Web Mercator 등각도법 — X, Y 모두 1/cos(φ) 적용
    let lat_rad = 2.0 * (particle.coordinate.latitude * std::f64::consts::PI / 20037508.34).exp().atan() - std::f64::consts::FRAC_PI_2;
    let cos_lat = lat_rad.cos();
    let new_longitude = particle.coordinate.longitude + interpolated_u * exaggeration / cos_lat;
    let new_latitude = particle.coordinate.latitude + interpolated_v * exaggeration / cos_lat;
    let new_pixel = to_pixel(new_longitude, new_latitude, extent, resolution);

    // 4. 보간 처리 후 다시 반사 또는 우회 여부 확인
    if should_reflect_or_detour(particle, mask_data, new_pixel, width as usize, height as usize, exaggeration) {
        update_particle_state(particle, mask_data, data_bounds, extent, resolution, size, life);
        return;  // 이미 반사 또는 우회가 이루어졌으면 여기서 종료
    }

    // 정상 수행
    particle.coordinate.longitude = new_longitude;
    particle.coordinate.latitude = new_latitude;
    particle.current_u = interpolated_u;
    particle.current_v = interpolated_v;

    particle.status = 0;

    update_particle_state(particle, mask_data, data_bounds, extent, resolution,size, life);
}*/

// 파티클 상태 업데이트 함수
fn update_particle_state(
    particle: &mut Particle,
    /*mask_data: &Vec<u8>,*/
    data_bounds: SimpleBounds,
    extent: SimpleBounds,
    resolution: f64,
    size: [f64; 2],
    life: i16,
    prev_count: i16,
    rng: &mut Xoshiro256PlusPlus,
) {
    if particle.prev_coordinats.len() >= prev_count as usize {
        particle.prev_coordinats.remove(0);
    }
    if particle.loop_count % 2 == 0 {
        particle.prev_coordinats.push(particle.coordinate);
    }
    particle.loop_count += 1;

    // 파티클이 수명이 다한 경우 다시 생성
    if should_respawn(particle) {
        let contain_bound = if extent.contains_bounds(data_bounds)  {extent} else {data_bounds};
        *particle = generate_particle(extent/*, mask_data*/, contain_bound, resolution, size, life, rng);
        particle.status = ParticleStatus::NO;
    }
}

fn interpolate_uv(
    particle: &mut Particle,
    rtree: &RTree<WeatherData>,
    interpolation_type:&String,
    extent: SimpleBounds,
    resolution: f64,
    size: [f64; 2],
) -> (f64, f64) {
    let i_type = interpolation_type.as_str();
    let rtree_ref = rtree;

    let (u, v) = match i_type {
        "IDW" => interpolate_by_inverse_distance_weighted(particle.coordinate, rtree_ref, Some(4), /*mask_data,*/ extent, resolution, size),
        /*"KRIGING" => interpolate_by_kriging(particle.coordinate, rtree_ref, Some(6), mask_data, extent, resolution, size),*/
        /*"BILINEAR" => interpolate_by_bilinear(particle.coordinate, rtree_ref, mask_data, extent, resolution, size),*/
        "BICUBIC" => interpolate_by_bicubic(particle.coordinate, rtree_ref, /*mask_data,*/ extent, resolution, size),
        "NEAREST" => interpolate(particle.coordinate, rtree_ref, extent, resolution, size),
        _ => (999.0, 999.0),  // 기본값 처리
    };
    (u, v)
}

fn detour_around_land(
    particle: &mut Particle,
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
    width: usize,
    height: usize,
    exaggeration: &f64
) {
    let (normal_x, normal_y) = calculate_normal_vector(pixel, mask_data, width, height);
    // 법선 벡터와 수직한 방향으로 이동 (우회)
    let mut detour_u = -normal_y * particle.current_u.abs(); // 속도의 크기는 유지하고 방향만 조정
    let mut detour_v = normal_x * particle.current_v.abs();

    let min_detour_speed = 0.2;  // 우회 시 최소 속도 설정
    let detour_speed = (detour_u.powi(2) + detour_v.powi(2)).sqrt();  // 현재 우회 속도 계산

    // 우회 속도가 너무 작으면 최소 속도로 설정
    if detour_speed < min_detour_speed && detour_speed != 0.0 {
        let scale_factor = min_detour_speed / detour_speed;
        detour_u *= scale_factor;
        detour_v *= scale_factor;
    }

    if detour_speed == 0.0 {
        detour_u = particle.current_u * 0.25;  // 기존 속도의 절반으로 기본 방향 설정
        detour_v = particle.current_v * 0.25;
    }

    particle.coordinate.longitude += detour_u * exaggeration;
    particle.coordinate.latitude += detour_v * exaggeration;

    let dot_product = particle.current_u * detour_u + particle.current_v * detour_v;

    // 두 벡터의 크기 계산
    let magnitude1 = (particle.current_u.powi(2) + particle.current_v.powi(2)).sqrt();
    let magnitude2 = (detour_u.powi(2) + detour_v.powi(2)).sqrt();

    // 벡터 사이 각도 계산 (라디안 값 반환)
    let angle_radians = (dot_product / (magnitude1 * magnitude2)).acos();

    // 라디안을 도로 변환해서 반환
    let degree = angle_radians.to_degrees();
    if degree < 10.0 || degree > 170.0{
        particle.status = ParticleStatus::RESPAWN;
    } else {
        particle.current_u = detour_u;
        particle.current_v = detour_v;

        particle.avoid_interpolation_frame_count = Some(1);

        //나주엥 다른거
        particle.status = ParticleStatus::RESPAWN;
    }
}

// 반사 처리 함수
fn handle_reflection(
    particle: &mut Particle,
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
    width: usize,
    height: usize,
    exaggeration: &f64
) {
    let (reflect_u, reflect_v) = calculate_reflection(particle, pixel, mask_data, width, height);
    particle.coordinate.longitude += reflect_u * exaggeration;
    particle.coordinate.latitude += reflect_v * exaggeration;
    particle.current_u = reflect_u;
    particle.current_v = reflect_v;

    particle.avoid_interpolation_frame_count = Some(7);

    particle.status = ParticleStatus::RESPAWN;
}

// 반사 및 우회를 결정하는 함수
fn should_reflect_or_detour(
    particle: &mut Particle,
    mask_data: &Vec<u8>,
    pixel: [f64; 2],
    width: usize,
    height: usize,
    exaggeration: &f64
) -> bool {
    let distance_to_land = calculate_distance_to_land(pixel, mask_data, width, height);
    if let Some(distance) = distance_to_land {
        if distance < 5.0 {
            particle.status = ParticleStatus::RESPAWN;
           return true;
        }  /*else if distance < 5.0 {
            // 매우 가까운 경우 반사
            handle_reflection(particle, pixel, mask_data, width, height, exaggeration);
            return true;
        } else if distance < 10.0 {
            // 가까운 경우 우회
            detour_around_land(particle, pixel, mask_data, width, height, exaggeration);
            return true;
        }*/
    }

    false
}

fn calculate_reflection(
    particle: &Particle,
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
    width: usize,
    height: usize
) -> (f64, f64) {
    let (normal_x, normal_y) = calculate_normal_vector(pixel, mask_data, width, height);
    let dot_product = particle.current_u * normal_x + particle.current_v * normal_y;

    let reflect_u = particle.current_u - 2.0 * dot_product * normal_x;
    let reflect_v = particle.current_v - 2.0 * dot_product * normal_y;

    // 반사된 속도 크기를 제한 (최대 속도를 설정)
    let max_speed = 0.2;
    let speed = (reflect_u.powi(2) + reflect_v.powi(2)).sqrt();
    if speed > max_speed {
        let scale_factor = max_speed / speed;
        return (reflect_u * scale_factor, reflect_v * scale_factor);
    }

    // 최소 속도 적용은 생략하여 반사만 먼저 적용해 봄
    (reflect_u, reflect_v)
}

fn should_reflect(mask_data: &Vec<u8>, pixel: [f64; 2], width: usize, height: usize) -> bool {
    let x = pixel[0].floor() as usize;
    let y = pixel[1].floor() as usize;

    if let Some(pixel_index) = calculate_pixel_index(x, y, width, height) {
        mask_data[pixel_index] == 0  // R 채널만 체크 (육지라면 true)
    } else {
        false
    }
}

fn should_respawn(particle: &Particle) -> bool {
    (particle.current_u.abs() < 0.001 && particle.current_v.abs() < 0.001) || particle.loop_count >= particle.life  || particle.status == ParticleStatus::RESPAWN
}

fn calculate_normal_vector(
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
    width: usize,
    height: usize,
) -> (f64, f64) {
    let mut radius = 1;
    let max_radius = 9;  // 반경을 단계적으로 확장

    while radius <= max_radius {
        let (normal_x, normal_y) = calculate_normal_in_radius(pixel, mask_data, width, height, radius);
        if normal_x != 0.0 || normal_y != 0.0 {
            // 육지가 발견되면 바로 반환
            return (normal_x, normal_y);
        }
        radius += 1;  // 반경을 4씩 확장하면서 탐색
    }

    (0.0, 0.0)  // 육지를 찾지 못했으면 기본값 반환
}

fn calculate_normal_in_radius(
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
    width: usize,
    height: usize,
    radius: usize
) -> (f64, f64) {
    let mut grad_x: f64 = 0.0;
    let mut grad_y: f64 = 0.0;

    let x = pixel[0].floor() as isize;
    let y = pixel[1].floor() as isize;

    // 주어진 반경 내의 좌표를 탐색
    for dx in -(radius as isize)..=radius as isize {
        for dy in -(radius as isize)..=radius as isize {
            let nx = (x + dx).clamp(0, width as isize - 1) as usize;
            let ny = (y + dy).clamp(0, height as isize - 1) as usize;

            if let Some(pixel_index) = calculate_pixel_index(nx, ny, width, height) {
                if mask_data[pixel_index] == 0 {
                    grad_x += dx as f64;
                    grad_y += dy as f64;
                }
            }
        }
    }

    let length = (grad_x.powi(2) + grad_y.powi(2)).sqrt();
    if length != 0.0 {
        (grad_x / length, grad_y / length)  // 정규화된 법선 벡터 반환
    } else {
        (0.0, 0.0)  // 육지가 없으면 (0, 0) 반환
    }
}

fn calculate_distance_to_land(
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
    width: usize,
    height: usize
) -> Option<f64> {
    let x = pixel[0].floor() as isize;
    let y = pixel[1].floor() as isize;
    let mut radius = 1;
    let max_radius = 8;  // 최대 반경 설정

    while radius <= max_radius {
        for dx in -(radius as isize)..=(radius as isize) {
            for dy in -(radius as isize)..=(radius as isize) {
                if dx.abs() != (radius as isize) && dy.abs() != (radius as isize) {
                    continue;
                }
                let nx = (x + dx).clamp(0, width as isize - 1) as usize;
                let ny = (y + dy).clamp(0, height as isize - 1) as usize;

                if let Some(pixel_index) = calculate_pixel_index(nx, ny, width, height) {
                    if mask_data[pixel_index] == 0 {
                        let distance = ((dx.pow(2) + dy.pow(2)) as f64).sqrt();
                        return Some(distance);  // 육지와의 거리 반환
                    }
                }
            }
        }
        radius *= 2;  // 기하급수적으로 반경을 확장
    }

    None  // 육지를 찾지 못한 경우
}


fn draw_particle (ctx: &CanvasRenderingContext2d, particle: &Particle, extent:SimpleBounds, resolution:f64, pixel:[f64;2], line_width:f64) {
    let speed = particle.get_speed();
    let base_color = calculate_color_from_speed_lerp(speed, Some(1.0));

    ctx.set_line_cap("round");
    ctx.begin_path();
    ctx.set_line_width(line_width);

    let prev_length = particle.prev_coordinats.len();
    for (prev_index, prev_coordinate) in particle.prev_coordinats.iter().enumerate() {
        let prev_pixel = to_pixel(prev_coordinate.longitude, prev_coordinate.latitude, extent, resolution);
        let prev_x = prev_pixel[0];
        let prev_y = prev_pixel[1];
        if prev_index == 0 {
            ctx.move_to(prev_x, prev_y);
        } else {
            let alpha = prev_index as f64 / prev_length as f64;
            let prev_color = calculate_color_from_speed_lerp(speed, Some(alpha));

            ctx.set_stroke_style(&JsValue::from_str(&prev_color));
            ctx.line_to(prev_x, prev_y);
            ctx.stroke();
            ctx.begin_path();
            ctx.move_to(prev_x, prev_y);
        }
    }
    ctx.line_to(pixel[0], pixel[1]);
    ctx.set_stroke_style(&JsValue::from_str(&base_color));
    ctx.stroke();

}

fn draw_vector (ctx: &CanvasRenderingContext2d, point:WeatherData, extent:SimpleBounds, resolution:f64, debug:bool) {
    let coordinate = point.coordinate;
    let lon = coordinate.longitude;
    let lat = coordinate.latitude;
    let pixel = to_pixel(lon, lat, extent, resolution);

    let rotate = point.rotate();
    let speed = point.speed();
    let knots = speed * 1.94384;
    ctx.save();

    ctx.translate(pixel[0], pixel[1]).expect("ctx translate failed");
    ctx.rotate(- (std::f64::consts::PI / 2.0) + rotate).expect("ctx rotate failed");
    draw_arrow(&ctx, speed).expect("draw arrow failed");

    ctx.restore();

    //draw_debug_vector(ctx, point.udata, point.vdata, lon, lat, pixel[0], pixel[1]);
}

fn rotate_point(x: f64, y: f64, angle_degrees: f64) -> (f64, f64) {
    // 각도를 라디안으로 변환
    let angle_radians = angle_degrees * (std::f64::consts::PI / 180.0);

    // cos(각도)와 sin(각도) 계산
    let cos_theta = angle_radians.cos();
    let sin_theta = angle_radians.sin();

    // 회전 행렬 적용
    let new_x = x * cos_theta - y * sin_theta;
    let new_y = x * sin_theta + y * cos_theta;

    (new_x, new_y)
}

fn draw_arrow(ctx: &CanvasRenderingContext2d, speed:f64,) -> Result<(), JsValue> {
    let dpi =  window().unwrap().device_pixel_ratio() * 96.0; // 일반적으로 96dpi로 계산
    // let pixels_per_cm = dpi / 2.54; // 1cm에 해당하는 픽셀 수
    let pixels_per_cm = dpi / 2.54; // 1cm에 해당하는 픽셀 수
    let arrow_length = (speed  * pixels_per_cm).min(pixels_per_cm * 1.5);
    let offset = /*arrow_length / 2.0 - 0.75*/0.0;
    let half_width = 1.5;  // 몸통 고정 너비

    let arrow_head_size = 9.0/*(5.0 / speed).clamp(1.5, 10.0)*/;   // 머리 고정 크기
    let arrow_tail_length =  (arrow_length - arrow_head_size).max(1.0);

    let width = half_width * 2.0;
    let curve_radius = 1.25;
    let adjusted_tail_length = arrow_tail_length - curve_radius;

    let gradient = ctx.create_linear_gradient(0.0, 0.0, -adjusted_tail_length, 0.0);
    gradient.add_color_stop(0.0, &*calculate_color_from_speed(speed, 0.9));
    gradient.add_color_stop(1.0, &*calculate_color_from_speed(speed, calculate_alpha_from_speed(speed)));

    ctx.set_fill_style(&gradient.into());
    ctx.set_stroke_style(&JsValue::from_str("#EAEAEA"));
    ctx.set_line_width(0.7);

    // 화살표 꼬리 (직사각형)
    ctx.begin_path();
    let left = rotate_point(offset, -6.0, -25.0);
    let right = rotate_point(offset, 6.0, 25.0);
    ctx.move_to(left.0, left.1/*width*(1.2 / speed).clamp(1.2, 1.8)*/); // 머리 좌측
    ctx.line_to(arrow_head_size + offset, 0.0); // 머리 끝점 (삼각형 끝, 가운데)
    ctx.line_to(right.0, right.1/*width*(1.2 / speed).clamp(1.2, 1.8)*/); // 머리 우측
    ctx.line_to(0.0 + offset, half_width); // 몸통 아래쪽 (머리와 연결)
    ctx.line_to(-adjusted_tail_length + offset, half_width); // 몸통 아래쪽 끝

    // 몸통과 곡선 연결
    ctx.quadratic_curve_to(
        -adjusted_tail_length - curve_radius + offset, // 제어점 X
        0.0,                                  // 제어점 Y
        -adjusted_tail_length + offset,                // 몸통 위쪽 끝 X
        -half_width,                          // 몸통 위쪽 끝 Y
    );

    // 화살표 머리로 다시 연결
    ctx.line_to(0.0 + offset, -half_width); // 몸통 위쪽 (머리와 연결)

    ctx.close_path();
    ctx.fill();
    ctx.stroke();

    Ok(())
}

fn draw_debug_vector(ctx: &CanvasRenderingContext2d, u:f64, v:f64, lon:f64, lat:f64, x:f64, y:f64) -> Result<(), JsValue> {
    ctx.set_text_align("center");
    let rotate = /*(std::f64::consts::PI / 2.0) - */v.atan2(u);
    let speed = (u * u + v * v).sqrt();

    let wgs84 = web_mercator_to_wgs84(lon, lat);

    /*let location_text = format!("Longitude: {:.2}, Latitude: {:.2}", wgs84.longitude, wgs84.latitude);
    ctx.set_font("11px Arial");
    ctx.set_fill_style(&JsValue::from_str("black"));
    ctx.fill_text(&location_text, x, y)?;*/

    // 회전 및 속도 출력
    let rotation_speed_text = format!("Rotation rad: {:.2} Rotation Deg: {:.2}° Speed: {:.2}",rotate, rotate.to_degrees(), speed);
    ctx.fill_text(&rotation_speed_text, x, y + 10.0)?;

    Ok(())
}

fn calculate_color_from_speed(speed: f64, alpha: f64) -> String {
    match speed {
        0.0..=0.26 => format!("rgba(127, 0, 127, {})", alpha),   // 보라색
        0.26..=0.51 => format!("rgba(3, 3, 202, {})", alpha),    // 파란색
        0.51..=0.76 => format!("rgba(0, 93, 0, {})", alpha),     // 초록색
        0.76..=1.01 => format!("rgba(233, 88, 0, {})", alpha),   // 주황색
        _ => format!("rgba(192, 0, 0, {})", alpha),              // 빨간색 (1.01 이상)
    }
}

fn calculate_alpha_from_speed(speed: f64) -> f64 {
    let min_alpha = 0.5;
    let max_alpha = 1.0;
    let threshold_speed = 0.26;

    if speed <= threshold_speed {
        max_alpha // 속도가 threshold_speed 이하일 때는 알파값을 1.0에 가깝게 유지
    } else {
        // 속도가 threshold_speed를 넘을 때 점차 알파값을 줄임 (0.3까지)
        max_alpha - (speed - threshold_speed) / (1.01 - threshold_speed) * (max_alpha - min_alpha)
    }
}

fn lerp_color(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f64, alpha:f64) -> String {
    let r = c1.0 as f64 + (c2.0 as f64 - c1.0 as f64) * t;
    let g = c1.1 as f64 + (c2.1 as f64 - c1.1 as f64) * t;
    let b = c1.2 as f64 + (c2.2 as f64 - c1.2 as f64) * t;

    format!("rgba({}, {}, {}, {})", r as u8, g as u8, b as u8, alpha)
}

// 속도에 따라 색상 계산 함수
fn calculate_color_from_speed_lerp(speed: f64, alpha:Option<f64>) -> String {
    let alpha = alpha.unwrap_or(1.0);
    if speed <= 0.26 {
        // 보라색 -> 파란색
        lerp_color((127, 0, 127), (3, 3, 202), speed / 0.26, alpha)
    } else if speed <= 0.51 {
        // 파란색 -> 초록색
        lerp_color((3, 3, 202), (0, 93, 0), (speed - 0.26) / (0.51 - 0.26), alpha)
    } else if speed <= 0.76 {
        // 초록색 -> 주황색
        lerp_color((0, 93, 0), (233, 88, 0), (speed - 0.51) / (0.76 - 0.51), alpha)
    } else if speed <= 1.01 {
        // 주황색 -> 빨간색
        lerp_color((233, 88, 0), (192, 0, 0), (speed - 0.76) / (1.01 - 0.76), alpha)
    } else {
        // 빨간색
        lerp_color((192, 0, 0), (192, 0, 0), 1.0, alpha)
    }
}

fn initialize_data_with_particle(data_array: Vec<(f64,f64,f64,f64)>, extent: SimpleBounds, resolution: f64, size:[f64;2], particle_count: Option<usize>, life:i16, rng: &mut Xoshiro256PlusPlus) -> (RTree<WeatherData>, SimpleBounds, Option<Vec<Particle>>) {
    let (rtree, simple_bounds) = initialize_data(data_array, extent, resolution);

    let particles = if let Some(particle_count) = particle_count {
        Some(generate_particles(extent, particle_count/*, mask_data*/, simple_bounds, resolution, size, life, rng))
    } else {
        None
    };

    (rtree, simple_bounds, particles)
}

fn initialize_data(data_array: Vec<(f64,f64,f64,f64)>, extent: SimpleBounds, resolution: f64) -> (RTree<WeatherData>, SimpleBounds) {
    let mut rtree = RTree::new();
    //log!("데이터 추출 전");
    let _ = for data in data_array.iter() {
        //log!("data {:?}", data);
        // 각 item이 객체라고 가정하고 필요한 필드를 추출
        let longitude = data.0;
        let latitude = data.1;
        let udata = data.2;
        let vdata = data.3;

        //log!("longitude {:?}, latitude {:?}, udata {:?}, vdata {:?}", longitude, latitude, udata, vdata);
        let coord = wgs84_to_web_mercator(longitude, latitude);
        let pixel = to_pixel(coord.longitude, coord.latitude, extent, resolution);

        let wd = WeatherData {
            coordinate: coord,
            udata,
            vdata,
        };
        rtree.insert(wd);
    };
    //log!("데이터 추출 종료");
    let bounds = rtree.root().envelope();
    let simple_bounds = SimpleBounds::from_aabb_point(&bounds);

    (rtree, simple_bounds)
}

fn generate_particles(extent:SimpleBounds,num_particles: usize/*,mask_data:&Vec<u8>*/, data_bounds:SimpleBounds, resolution: f64, size:[f64;2], life:i16, rng: &mut Xoshiro256PlusPlus) -> Vec<Particle> {
    let mut particles = Vec::new();
    let contain_bound = if extent.contains_bounds(data_bounds)  {extent} else {data_bounds};
    for _ in 0..num_particles {
        particles.push(generate_particle(extent,/*mask_data,*/ contain_bound, resolution, size, life, rng));
    }

    particles
}

fn generate_particle(extent:SimpleBounds/*, mask_data:&Vec<u8>*/, data_bounds:SimpleBounds, resolution: f64, size:[f64;2], life:i16, rng: &mut Xoshiro256PlusPlus) -> Particle {
    let width = size[0];
    let height = size[1];

    // 역전된 bounds 또는 크기 0인 경우 무한루프 방지: fallback 파티클 반환
    if width <= 0.0 || height <= 0.0 || data_bounds.min_x >= data_bounds.max_x || data_bounds.min_y >= data_bounds.max_y {
        let cx = (data_bounds.min_x + data_bounds.max_x) / 2.0;
        let cy = (data_bounds.min_y + data_bounds.max_y) / 2.0;
        let coordinate = Coordinate { longitude: cx, latitude: cy };
        return Particle {
            coordinate,
            original_coordinate: coordinate,
            prev_coordinats: Vec::new(),
            current_u: 0.0,
            current_v: 0.0,
            loop_count: 0,
            avoid_interpolation_frame_count: None,
            status: ParticleStatus::NO,
            life: random_within_percent(life, rng),
            prev_coordinate: coordinate,
        };
    }

    const MAX_ATTEMPTS: usize = 1000;
    for _ in 0..MAX_ATTEMPTS {
        let x = rng.gen_range(0..width as usize);
        let y = rng.gen_range(0..height as usize);
        let [longitude, latitude] = to_coord([x as f64, y as f64], extent, resolution);
        let coordinate = Coordinate { longitude, latitude};

        if data_bounds.contains_coordinate(coordinate) {
            return Particle {
                coordinate,
                original_coordinate: coordinate,
                prev_coordinats: Vec::new(),
                current_u: 0.0,
                current_v: 0.0,
                loop_count: 0,
                avoid_interpolation_frame_count: None,
                status: ParticleStatus::NO,
                life: random_within_percent(life, rng),
                prev_coordinate: coordinate,
            };
        }
    }

    // fallback: data_bounds 중심
    let cx = (data_bounds.min_x + data_bounds.max_x) / 2.0;
    let cy = (data_bounds.min_y + data_bounds.max_y) / 2.0;
    let coordinate = Coordinate { longitude: cx, latitude: cy };
    Particle {
        coordinate,
        original_coordinate: coordinate,
        prev_coordinats: Vec::new(),
        current_u: 0.0,
        current_v: 0.0,
        loop_count: 0,
        avoid_interpolation_frame_count: None,
        status: ParticleStatus::NO,
        life: random_within_percent(life, rng),
        prev_coordinate: coordinate,
    }
}

/// mesh 경로용 파티클 생성: mesh_bounds 내 랜덤 좌표에서 삼각형 안에 들어갈 때까지 재시도
fn generate_particle_mesh(
    mesh_bounds: SimpleBounds,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    life: i16,
    rng: &mut Xoshiro256PlusPlus,
) -> Particle {
    const MAX_TRIES: usize = 10;
    for _ in 0..MAX_TRIES {
        let x = rng.gen_range(mesh_bounds.min_x..mesh_bounds.max_x);
        let y = rng.gen_range(mesh_bounds.min_y..mesh_bounds.max_y);

        if mesh_interpolate_uv(x, y, rtree, nodes).is_some() {
            let coordinate = Coordinate { longitude: x, latitude: y };
            return Particle {
                coordinate,
                original_coordinate: coordinate,
                prev_coordinats: Vec::new(),
                current_u: 0.0,
                current_v: 0.0,
                loop_count: 0,
                avoid_interpolation_frame_count: None,
                status: ParticleStatus::NO,
                life: random_within_percent(life, rng),
                prev_coordinate: coordinate,
            };
        }
    }

    // fallback: 삼각형 직접 선택 (결정적 중심점 대신 랜덤 삼각형 내부 좌표)
    if let Some(p) = spawn_in_random_triangle_global(rtree, nodes, life, rng) {
        return p;
    }
    // 삼각형이 전혀 없는 극단적 경우: mesh_bounds 내 랜덤 좌표
    let rx = rng.gen_range(mesh_bounds.min_x..mesh_bounds.max_x);
    let ry = rng.gen_range(mesh_bounds.min_y..mesh_bounds.max_y);
    let coordinate = Coordinate { longitude: rx, latitude: ry };
    Particle {
        coordinate,
        original_coordinate: coordinate,
        prev_coordinats: Vec::new(),
        current_u: 0.0,
        current_v: 0.0,
        loop_count: 0,
        avoid_interpolation_frame_count: None,
        status: ParticleStatus::NO,
        life: random_within_percent(life, rng),
        prev_coordinate: coordinate,
    }
}

/// Phase 2 Step 6: 셀 격자 카운트 빌드
fn build_cell_counts(particles: &[Particle], viewport: &Viewport) -> Vec<u32> {
    let total_cells = viewport.grid_cols * viewport.grid_rows;
    let mut counts = vec![0u32; total_cells];
    for p in particles {
        let (sx, sy) = viewport.geo_to_screen(p.coordinate.longitude, p.coordinate.latitude);
        if sx >= 0.0 && sx < viewport.canvas_w && sy >= 0.0 && sy < viewport.canvas_h {
            let (col, row) = viewport.screen_to_cell(sx, sy);
            let idx = row * viewport.grid_cols + col;
            if idx < total_cells {
                counts[idx] += 1;
            }
        }
    }
    counts
}

/// Phase 2 Step 6: 최소 밀도 셀 탐색
fn find_sparsest_cell(counts: &[u32], cols: usize, rows: usize) -> (usize, usize) {
    let mut min_count = u32::MAX;
    let mut min_col = 0;
    let mut min_row = 0;
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if counts[idx] < min_count {
                min_count = counts[idx];
                min_col = col;
                min_row = row;
            }
        }
    }
    (min_col, min_row)
}

/// 메시 커버리지 셀 마스크 빌드: 각 셀의 중심+모서리를 샘플링하여 메시 존재 여부 판정
fn build_mesh_cell_mask(
    viewport: &Viewport,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
) -> Vec<bool> {
    let total = viewport.grid_cols * viewport.grid_rows;
    let mut mask = vec![false; total];

    for row in 0..viewport.grid_rows {
        for col in 0..viewport.grid_cols {
            let idx = row * viewport.grid_cols + col;
            // 셀 중심 + 4개 모서리 샘플링
            let samples = [
                ((col as f64 + 0.5) * CELL_SIZE, (row as f64 + 0.5) * CELL_SIZE),
                (col as f64 * CELL_SIZE, row as f64 * CELL_SIZE),
                ((col as f64 + 1.0) * CELL_SIZE, row as f64 * CELL_SIZE),
                (col as f64 * CELL_SIZE, (row as f64 + 1.0) * CELL_SIZE),
                ((col as f64 + 1.0) * CELL_SIZE, (row as f64 + 1.0) * CELL_SIZE),
            ];
            for (sx, sy) in &samples {
                let (gx, gy) = viewport.screen_to_geo(*sx, *sy);
                if mesh_interpolate_uv(gx, gy, rtree, nodes).is_some() {
                    mask[idx] = true;
                    break;
                }
            }
        }
    }
    mask
}

/// 메시 커버리지 마스크를 고려한 최소 밀도 셀 탐색 (Reservoir Sampling으로 랜덤 타이브레이킹)
fn find_sparsest_valid_cell(counts: &[u32], mask: &[bool], cols: usize, rows: usize, rng: &mut Xoshiro256PlusPlus) -> Option<(usize, usize)> {
    let mut min_count = u32::MAX;
    let mut min_col = 0;
    let mut min_row = 0;
    let mut tie_count: usize = 0;
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            if idx < mask.len() && mask[idx] && idx < counts.len() {
                if counts[idx] < min_count {
                    // 새로운 최솟값 발견 → 리셋
                    min_count = counts[idx];
                    min_col = col;
                    min_row = row;
                    tie_count = 1;
                } else if counts[idx] == min_count {
                    // 동률 → Reservoir Sampling: 1/tie_count 확률로 교체
                    tie_count += 1;
                    if rng.gen_range(0..tie_count) == 0 {
                        min_col = col;
                        min_row = row;
                    }
                }
            }
        }
    }
    if tie_count > 0 { Some((min_col, min_row)) } else { None }
}

/// 삼각형 내부 균등 분포 랜덤 좌표 생성
/// 공식: P = (1 - √r1)A + √r1(1 - r2)B + √r1·r2·C
fn random_point_in_triangle(tri: &TriangleRef, nodes: &[MeshNode], rng: &mut Xoshiro256PlusPlus) -> (f64, f64) {
    let (i0, i1, i2) = (tri.indices[0] as usize, tri.indices[1] as usize, tri.indices[2] as usize);
    let r1_sqrt = rng.gen_range(0.0..1.0_f64).sqrt();
    let r2: f64 = rng.gen_range(0.0..1.0);
    let x = (1.0 - r1_sqrt) * nodes[i0].x + r1_sqrt * (1.0 - r2) * nodes[i1].x + r1_sqrt * r2 * nodes[i2].x;
    let y = (1.0 - r1_sqrt) * nodes[i0].y + r1_sqrt * (1.0 - r2) * nodes[i1].y + r1_sqrt * r2 * nodes[i2].y;
    (x, y)
}

/// 뷰포트 내 삼각형에서 면적 가중 랜덤 선택 → 삼각형 내부 스폰
/// 캐시된 누적 면적 배열에서 binary search로 O(log N) 선택
fn spawn_in_random_triangle_viewport(
    cache: &VisibleTriangleCache,
    nodes: &[MeshNode],
    life: i16,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<Particle> {
    if cache.triangles.is_empty() || cache.total_area <= 0.0 {
        return None;
    }

    // 누적 면적 배열에서 binary search로 삼각형 선택
    let r = rng.gen_range(0.0..cache.total_area);
    let idx = match cache.cumulative_areas.binary_search_by(|a| {
        a.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(i) => i,
        Err(i) => i.min(cache.triangles.len() - 1),
    };

    let tri = &cache.triangles[idx];
    let (x, y) = random_point_in_triangle(tri, nodes, rng);
    let coordinate = Coordinate { longitude: x, latitude: y };

    Some(Particle {
        coordinate,
        original_coordinate: coordinate,
        prev_coordinats: Vec::new(),
        current_u: 0.0,
        current_v: 0.0,
        loop_count: 0,
        avoid_interpolation_frame_count: None,
        status: ParticleStatus::NO,
        life: random_within_percent(life, rng),
        prev_coordinate: coordinate,
    })
}

/// 면적 가중 삼각형 스폰 + 밀도 rejection
/// 최대 5회 시도, 실패 시 마지막 시도 결과를 그대로 반환
fn spawn_balanced_particle(
    viewport: &Viewport,
    cache: &VisibleTriangleCache,
    nodes: &[MeshNode],
    life: i16,
    counts: &mut Vec<u32>,
    max_per_cell: u32,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<Particle> {
    const MAX_ATTEMPTS: usize = 5;

    let mut last_particle: Option<Particle> = None;

    for _ in 0..MAX_ATTEMPTS {
        let p = match spawn_in_random_triangle_viewport(cache, nodes, life, rng) {
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

/// 메시 전체에서 면적 가중 랜덤 삼각형 선택 → 삼각형 내부 스폰 (100% 성공)
fn spawn_in_random_triangle_global(
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    life: i16,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<Particle> {
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
        if rng.gen_range(0.0..cumulative_area) < area {
            selected = Some(tri);
        }
    }
    let tri = selected?;
    let (x, y) = random_point_in_triangle(tri, nodes, rng);
    let coordinate = Coordinate { longitude: x, latitude: y };
    Some(Particle {
        coordinate,
        original_coordinate: coordinate,
        prev_coordinats: Vec::new(),
        current_u: 0.0,
        current_v: 0.0,
        loop_count: 0,
        avoid_interpolation_frame_count: None,
        status: ParticleStatus::NO,
        life: random_within_percent(life, rng),
        prev_coordinate: coordinate,
    })
}

/// Phase 2 Step 6: 셀 내 랜덤 좌표에 파티클 생성
/// 3회 실패 시 8-인접 셀 시도, 전부 실패 시 삼각형 직접 선택 폴백
fn spawn_in_cell(
    col: usize,
    row: usize,
    viewport: &Viewport,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    life: i16,
    counts: &mut Vec<u32>,
    cache: &VisibleTriangleCache,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<Particle> {
    // 먼저 지정 셀에서 3회 시도
    if let Some(p) = try_spawn_in_cell(col, row, viewport, rtree, nodes, life, rng) {
        let idx = row * viewport.grid_cols + col;
        if idx < counts.len() {
            counts[idx] += 1;
        }
        return Some(p);
    }
    // 8-인접 셀 시도
    let deltas: [(isize, isize); 8] = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)];
    for (dc, dr) in &deltas {
        let nc = col as isize + dc;
        let nr = row as isize + dr;
        if nc < 0 || nr < 0 || nc >= viewport.grid_cols as isize || nr >= viewport.grid_rows as isize {
            continue;
        }
        let nc = nc as usize;
        let nr = nr as usize;
        if let Some(p) = try_spawn_in_cell(nc, nr, viewport, rtree, nodes, life, rng) {
            let idx = nr * viewport.grid_cols + nc;
            if idx < counts.len() {
                counts[idx] += 1;
            }
            return Some(p);
        }
    }
    // 셀+인접 셀 모두 실패 → 뷰포트 내 삼각형 직접 선택 폴백 (100% 성공)
    spawn_in_random_triangle_viewport(cache, nodes, life, rng)
}

/// 셀 내 랜덤 좌표에서 파티클 생성 시도 (최대 3회)
fn try_spawn_in_cell(
    col: usize,
    row: usize,
    viewport: &Viewport,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    life: i16,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<Particle> {
    let sx_min = col as f64 * CELL_SIZE;
    let sy_min = row as f64 * CELL_SIZE;
    let sx_max = (sx_min + CELL_SIZE).min(viewport.canvas_w);
    let sy_max = (sy_min + CELL_SIZE).min(viewport.canvas_h);

    for _ in 0..3 {
        let sx = rng.gen_range(sx_min..sx_max);
        let sy = rng.gen_range(sy_min..sy_max);
        let (gx, gy) = viewport.screen_to_geo(sx, sy);
        if mesh_interpolate_uv(gx, gy, rtree, nodes).is_some() {
            let coordinate = Coordinate { longitude: gx, latitude: gy };
            return Some(Particle {
                coordinate,
                original_coordinate: coordinate,
                prev_coordinats: Vec::new(),
                current_u: 0.0,
                current_v: 0.0,
                loop_count: 0,
                avoid_interpolation_frame_count: None,
                status: ParticleStatus::NO,
                life: random_within_percent(life, rng),
                prev_coordinate: coordinate,
            });
        }
    }
    None
}

/// 뷰포트 우선 리스폰 폴백: 뷰포트 내 삼각형 직접 선택 → 메시 전체 삼각형 폴백
fn spawn_in_viewport_fallback(
    cache: &VisibleTriangleCache,
    rtree: &RTree<TriangleRef>,
    nodes: &[MeshNode],
    mesh_bounds: SimpleBounds,
    life: i16,
    cell_counts: &mut Vec<u32>,
    rng: &mut Xoshiro256PlusPlus,
) -> Particle {
    // 뷰포트 내 삼각형 직접 선택 (100% 성공)
    if let Some(p) = spawn_in_random_triangle_viewport(cache, nodes, life, rng) {
        return p;
    }
    // 뷰포트에 삼각형이 없으면 메시 전체에서 삼각형 직접 선택
    if let Some(p) = spawn_in_random_triangle_global(rtree, nodes, life, rng) {
        return p;
    }
    // 극단적 폴백
    generate_particle_mesh(mesh_bounds, rtree, nodes, life, rng)
}

/// 잔여 수명이 가장 짧은 파티클부터 제거
fn remove_shortest_lived(particles: &mut Vec<Particle>, count: usize) {
    if count == 0 || particles.is_empty() {
        return;
    }
    // 잔여 수명(life - loop_count) 기준 인덱스 정렬
    let mut indices: Vec<usize> = (0..particles.len()).collect();
    indices.sort_by_key(|&i| (particles[i].life - particles[i].loop_count) as i32);
    // 앞에서 count개의 인덱스를 선택해서 제거 (뒤에서부터 제거해야 인덱스 안 깨짐)
    let to_remove: Vec<usize> = indices.into_iter().take(count).collect();
    let mut remove_set = vec![false; particles.len()];
    for idx in to_remove {
        remove_set[idx] = true;
    }
    let mut i = 0;
    particles.retain(|_| {
        let keep = !remove_set[i];
        i += 1;
        keep
    });
}

pub fn random_within_percent(input: i16, rng: &mut Xoshiro256PlusPlus) -> i16 {
    // ±30% 범위 계산
    let range = (input as f32 * 0.3).abs() as i16;
    let min_value = input - range;
    let max_value = input + range;

    let random_float: f32 = rng.gen_range(0.0..=1.0);

    // min_value와 max_value 사이의 랜덤 값 생성
    min_value + ((max_value - min_value) as f32 * random_float).round() as i16
}
fn is_masked(pixel: [f64;2], mask_data: &Vec<u8>, width: f64, height: f64) -> bool {
    let x = pixel[0] as usize;
    let y = pixel[1] as usize;

    // 범위 체크
    if x >= width as usize || y >= height as usize {
        return true; // 범위를 벗어나면 육지로 처리
    }

    let pixel_index = (y * width as usize + x) * 4;
    let result = mask_data[pixel_index];
    /*log!("x {:?}, y {:?}, pixel_index {:?}, mask_data[pixel_index] {:?}", x, y, pixel_index, result);
    log!("mask_data length: {:?}, expected: {:?}", mask_data.len(), (width as usize * height as usize));*/
    result == 0  // 육지라면 true
}

// 유클리드 거리 계산 함수 (WeatherData 기준)
fn euclidean_distance(p1: &WeatherData, p2: &Coordinate) -> f64 {
    let dx = p1.coordinate.longitude - p2.longitude;
    let dy = p1.coordinate.latitude - p2.latitude;
    (dx * dx + dy * dy).sqrt()
}

// 클러스터의 중심 좌표 계산
fn calculate_centroid(points: &[WeatherData]) -> Coordinate {
    let sum_x: f64 = points.iter().map(|p| p.coordinate.longitude).sum();
    let sum_y: f64 = points.iter().map(|p| p.coordinate.latitude).sum();
    let count = points.len() as f64;

    Coordinate {
        longitude: sum_x / count,
        latitude: sum_y / count,
    }
}

fn cluster_labels_to_clusters(points: &[WeatherData], cluster_labels: &[Option<usize>]) -> HashMap<usize, Vec<WeatherData>> {
    let mut clusters: HashMap<usize, Vec<WeatherData>> = HashMap::new();

    // 클러스터별로 WeatherData를 그룹화
    for (i, label) in cluster_labels.iter().enumerate() {
        if let Some(cluster_id) = label {
            clusters.entry(*cluster_id).or_insert_with(Vec::new).push(points[i]);
        }
    }
    clusters
}

// 클러스터별 대표 WeatherData 추출 함수
fn extract_representative_weather_data(
    points: &Vec<WeatherData>,
    cluster_labels: &[Option<usize>]
) -> HashMap<usize, Vec<WeatherData>> {
    let clusters: HashMap<usize, Vec<WeatherData>> = cluster_labels_to_clusters(points, cluster_labels);

    // 클러스터별 대표 WeatherData 계산
    let mut representative_points: HashMap<usize, Vec<WeatherData>> = HashMap::new();
    for (cluster_id, cluster_points) in clusters.iter() {
        let centroid = calculate_centroid(cluster_points); // 중심 계산
        // 중심에 가까운 순서대로 정렬
        let mut sorted_points = cluster_points.clone();
        sorted_points.sort_by(|a, b| {
            euclidean_distance(a, &centroid)
                .partial_cmp(&euclidean_distance(b, &centroid))
                .unwrap_or(Ordering::Equal)
        });

        // 가까운 순서대로 'count'만큼의 포인트를 가져옴
        let representatives: Vec<WeatherData> = sorted_points
            .iter()
            .take(1) // 가까운 순서대로 'count' 개수만큼 선택
            .cloned()
            .collect();
        representative_points.insert(*cluster_id, representatives);
    }

    representative_points
}

// 클러스터에서 일정 개수의 WeatherData를 랜덤으로 샘플링하는 함수
pub fn sample_weather_data_from_clusters(
    points: &[WeatherData],
    cluster_labels: &[Option<usize>],
    sample_size: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> HashMap<usize, Vec<WeatherData>> {
    let clusters: HashMap<usize, Vec<WeatherData>> = cluster_labels_to_clusters(points, cluster_labels);

    // 클러스터별로 랜덤 샘플링
    let mut sampled_points: HashMap<usize, Vec<WeatherData>> = HashMap::new();
    for (cluster_id, cluster_points) in clusters.iter() {
        let sample_size = sample_size.min(cluster_points.len()); // 클러스터의 포인트보다 큰 샘플은 방지
        let random_indices = get_random_index(cluster_points.len(), sample_size, rng); // 랜덤 인덱스 생성

        let sample: Vec<WeatherData> = random_indices
            .iter()
            .map(|&i| cluster_points[i].clone()) // 인덱스를 기반으로 포인트 복사
            .collect();

        sampled_points.insert(*cluster_id, sample);
    }

    sampled_points
}

// 랜덤 인덱스를 생성하는 함수
pub fn get_random_index(max: usize, count: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        indices.push(rng.gen_range(0..max));
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{MeshNode, TriangleRef};

    fn make_viewport() -> Viewport {
        // 800x600 canvas, Mercator extent [0, 0, 800, 600]
        let extent = SimpleBounds { min_x: 0.0, min_y: 0.0, max_x: 800.0, max_y: 600.0 };
        Viewport::new(800.0, 600.0, extent)
    }

    #[test]
    fn test_viewport_geo_to_screen() {
        let vp = make_viewport();
        // 좌상단 (0, 600) → screen (0, 0)
        let (sx, sy) = vp.geo_to_screen(0.0, 600.0);
        assert!((sx - 0.0).abs() < 1e-6);
        assert!((sy - 0.0).abs() < 1e-6);
        // 우하단 (800, 0) → screen (800, 600)
        let (sx, sy) = vp.geo_to_screen(800.0, 0.0);
        assert!((sx - 800.0).abs() < 1e-6);
        assert!((sy - 600.0).abs() < 1e-6);
        // 중앙 (400, 300) → screen (400, 300)
        let (sx, sy) = vp.geo_to_screen(400.0, 300.0);
        assert!((sx - 400.0).abs() < 1e-6);
        assert!((sy - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_screen_to_geo() {
        let vp = make_viewport();
        // screen (400, 300) → geo (400, 300)
        let (gx, gy) = vp.screen_to_geo(400.0, 300.0);
        assert!((gx - 400.0).abs() < 1e-6);
        assert!((gy - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_roundtrip() {
        let vp = make_viewport();
        let (gx, gy) = (123.4, 456.7);
        let (sx, sy) = vp.geo_to_screen(gx, gy);
        let (gx2, gy2) = vp.screen_to_geo(sx, sy);
        assert!((gx - gx2).abs() < 1e-6);
        assert!((gy - gy2).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_compute_target_count() {
        let vp = make_viewport();
        let target = vp.compute_target_count();
        // 800*600 / 50*50 = 192 → clamped to MIN_PARTICLES=1000
        assert_eq!(target, MIN_PARTICLES);

        // 큰 캔버스
        let big_vp = Viewport::new(2000.0, 2000.0, SimpleBounds { min_x: 0.0, min_y: 0.0, max_x: 2000.0, max_y: 2000.0 });
        let target = big_vp.compute_target_count();
        // 2000*2000 / 50*50 = 1600, within [1000, 5000]
        assert_eq!(target, 1600);
    }

    #[test]
    fn test_viewport_screen_to_cell() {
        let vp = make_viewport();
        // (0, 0) → cell (0, 0)
        assert_eq!(vp.screen_to_cell(0.0, 0.0), (0, 0));
        // (49, 49) → cell (0, 0)
        assert_eq!(vp.screen_to_cell(49.0, 49.0), (0, 0));
        // (50, 50) → cell (1, 1)
        assert_eq!(vp.screen_to_cell(50.0, 50.0), (1, 1));
    }

    #[test]
    fn test_serialize_screen_format_size() {
        let vp = make_viewport();
        let coord = Coordinate { longitude: 100.0, latitude: 200.0 };
        let particle = Particle {
            coordinate: coord,
            original_coordinate: coord,
            prev_coordinats: Vec::new(),
            current_u: 0.5,
            current_v: 0.3,
            loop_count: 10,
            avoid_interpolation_frame_count: None,
            status: ParticleStatus::OK,
            life: 100,
            prev_coordinate: Coordinate { longitude: 99.0, latitude: 199.0 },
        };
        let buf = particle.serialize_screen_format(&vp);
        assert_eq!(buf.len(), 40);

        // speed 검증
        let speed = f32::from_le_bytes(buf[32..36].try_into().unwrap());
        let expected_speed = (0.5f64.powi(2) + 0.3f64.powi(2)).sqrt() as f32;
        assert!((speed - expected_speed).abs() < 1e-5);

        // life_ratio 검증
        let life_ratio = f32::from_le_bytes(buf[36..40].try_into().unwrap());
        let expected_ratio = (100 - 10) as f32 / 100.0;
        assert!((life_ratio - expected_ratio).abs() < 1e-5);
    }

    #[test]
    fn test_build_cell_counts() {
        let vp = make_viewport();
        let coord1 = Coordinate { longitude: 25.0, latitude: 575.0 }; // screen ~(25, 25) → cell(0,0)
        let coord2 = Coordinate { longitude: 75.0, latitude: 575.0 }; // screen ~(75, 25) → cell(1,0)
        let particles = vec![
            Particle {
                coordinate: coord1,
                original_coordinate: coord1,
                prev_coordinats: Vec::new(),
                current_u: 0.0, current_v: 0.0,
                loop_count: 0, avoid_interpolation_frame_count: None,
                status: ParticleStatus::OK, life: 100,
                prev_coordinate: coord1,
            },
            Particle {
                coordinate: coord2,
                original_coordinate: coord2,
                prev_coordinats: Vec::new(),
                current_u: 0.0, current_v: 0.0,
                loop_count: 0, avoid_interpolation_frame_count: None,
                status: ParticleStatus::OK, life: 100,
                prev_coordinate: coord2,
            },
        ];
        let counts = build_cell_counts(&particles, &vp);
        assert_eq!(counts[0], 1); // cell(0,0)
        assert_eq!(counts[1], 1); // cell(1,0)
    }

    #[test]
    fn test_find_sparsest_cell() {
        let mut counts = vec![3, 1, 5, 2, 0, 4];
        let (col, row) = find_sparsest_cell(&counts, 3, 2);
        // index 4 (row=1, col=1) has count 0
        assert_eq!((col, row), (1, 1));
    }

    #[test]
    fn test_remove_shortest_lived() {
        let make_p = |life: i16, loop_count: i16| -> Particle {
            let coord = Coordinate { longitude: 0.0, latitude: 0.0 };
            Particle {
                coordinate: coord, original_coordinate: coord,
                prev_coordinats: Vec::new(),
                current_u: 0.0, current_v: 0.0,
                loop_count, avoid_interpolation_frame_count: None,
                status: ParticleStatus::OK, life,
                prev_coordinate: coord,
            }
        };
        let mut particles = vec![
            make_p(100, 90),  // 잔여 10
            make_p(100, 50),  // 잔여 50
            make_p(100, 99),  // 잔여 1
            make_p(100, 70),  // 잔여 30
        ];
        remove_shortest_lived(&mut particles, 2);
        assert_eq!(particles.len(), 2);
        // 잔여가 가장 짧은 2개(잔여 1, 잔여 10) 제거됨
        // 남은 것: 잔여 50, 잔여 30
        let remaining: Vec<i16> = particles.iter().map(|p| p.life - p.loop_count).collect();
        assert!(remaining.contains(&50));
        assert!(remaining.contains(&30));
    }
}
