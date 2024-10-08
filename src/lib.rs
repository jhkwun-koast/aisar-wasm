mod coordinate;
mod tiling_system;
mod tile;
mod util;
mod interpolate;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};
use js_sys::{Reflect, Uint8Array};
use rstar::{RTree, AABB};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use tiling_system::{ TilingSystem};
use getrandom::getrandom;
use web_sys::{console, js_sys, window, HtmlCanvasElement, CanvasRenderingContext2d};

use crate::coordinate::{web_mercator_to_wgs84, wgs84_to_web_mercator, Coordinate, WeatherData};
use crate::interpolate::{interpolate_by_bicubic, interpolate_by_inverse_distance_weighted, interpolate_by_kriging};
use crate::tile::Tile;
use crate::util::{calculate_pixel_index, generate_random_pixel, to_coord, to_pixel};

macro_rules! log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()));
}

#[derive(Deserialize)]
pub struct CurrentVectorWrapperOption {
    data: Vec<(f64, f64, f64, f64)>,
    mask_data: Vec<u8>,
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
    debug: bool
}

#[derive(Deserialize, Debug)]
pub struct CurrentFlowWrapperOption {
    particle_count:usize,
    data: Vec<(f64, f64, f64, f64)>,
    interpolation_type:String,
    extent: [f64; 4],
    resolution: f64,
    size: [f64; 2],
    mask_data: Vec<u8>
}

#[derive(Deserialize)]
pub struct DrawCurrentFlowCanvasOption {
    line_width: f64,
    life: i16,
    exaggeration: f64
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
        self.containsCoordinate(point.coordinate)
    }

    pub fn containsCoordinate(&self, coordinate: Coordinate) -> bool {
        let longitude = coordinate.longitude;
        let latitude = coordinate.latitude;

        longitude >= self.min_x && longitude <= self.max_x && latitude >= self.min_y && latitude <= self.max_y
    }
}

#[wasm_bindgen]
pub struct CurrentVectorWrapper {
    //tiling_system: Rc<RefCell<TilingSystem>>,  // Rc<RefCell<TilingSystem>>로 감쌈
    rtree: Rc<RefCell<RTree<WeatherData>>>,
    bounds: Rc<RefCell<SimpleBounds>>,
    mask_data: Vec<u8>
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
    let mask_data_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
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
    let mask_data = data_vec[offset..offset + mask_data_length].to_vec();
    offset += mask_data_length;
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
        mask_data,
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
        let mask_data = opts.mask_data;
        let extent = opts.extent;
        let resolution = opts.resolution;
        let size = opts.size;
        let (rtree, bounds, _) = initialize_data(data, &mask_data, SimpleBounds::from_extent(extent), resolution, size, None);
        //log!("rtree {:?}, bounds {:?}", rtree, bounds);
        CurrentVectorWrapper {
            rtree: Rc::new(RefCell::new(rtree)),
            bounds: Rc::new(RefCell::new(bounds)),
            mask_data
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

    pub fn drawCanvasCurrentVecor(&self, options:JsValue) -> Result<HtmlCanvasElement, JsValue> {
        let opts: DrawCurrentVectorCanvasOption = serde_wasm_bindgen::from_value(options)?;
        let size = opts.size;
        let raw_extent = opts.extent;
        let resolution = opts.resolution;
        let sampling_type = opts.sampling_type.as_str();
        let systematic_value = opts.systematic_value;
        let proportional_value = opts.proportional_value;
        let dbscan_eps_value = opts.dbscan_eps_value;
        let dbscan_minpoint_value = opts.dbscan_minpoint_value;
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
        ctx.scale(1.0, 1.0);
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
                let random_indices = get_random_index(wd_group.len(), sample_size); // 랜덤 인덱스 생성

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Particle {
    coordinate: Coordinate,
    original_coordinate: Coordinate,
    prev_coordinats: Vec<Coordinate>,
    current_u: f64,
    current_v: f64,
    loop_count: i16,
    avoid_interpolation_frame_count: Option<i16>,
    status: i8
}

impl Particle {
    pub fn get_speed(&self) -> f64 {
        (self.current_u.powi(2) + self.current_v.powi(2)).sqrt()
    }
}

fn binary_to_current_flow_wrapper_option(binary_data: Uint8Array) -> CurrentFlowWrapperOption {
    let data_vec = binary_data.to_vec();
    let mut offset = 0;

    // 1. dataArrayLength와 maskDataLength 읽기
    let data_array_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mask_data_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
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

    // 3. mask_data 해석
    let mask_data = data_vec[offset..offset + mask_data_length].to_vec();
    offset += mask_data_length;

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

    // 8. interpolation_type 해석 (문자열)
    let string_length = u32::from_le_bytes(data_vec[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let string_bytes = &data_vec[offset..offset + string_length];
    let interpolation_type = String::from_utf8(string_bytes.to_vec()).unwrap();
    offset += string_length;

    // CurrentVectorWrapperOption 구조체 반환
    CurrentFlowWrapperOption {
        data,
        mask_data,
        extent,
        resolution,
        size,
        particle_count,
        interpolation_type,
    }
}

#[wasm_bindgen]
pub struct CurrentFlowWrapper {
    tiling_system: Rc<RefCell<TilingSystem>>,  // Rc<RefCell<TilingSystem>>로 감쌈
    particles: Vec<Particle>,
    particle_count: usize,
    //per_frame_control_particle_count: usize,
    rendering_count: usize,
    interpolation_type: String,
    extent: [f64; 4],
    resolution: f64,
    size: [f64; 2],
    depth: usize,
    mask_data: Vec<u8>
}

#[wasm_bindgen]
impl CurrentFlowWrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(binary_data:Uint8Array ) -> CurrentFlowWrapper {
        let opts: CurrentFlowWrapperOption = binary_to_current_flow_wrapper_option(binary_data);

        let particle_count = opts.particle_count;
        let interpolation_type = opts.interpolation_type;
        let extent = opts.extent;
        let resolution = opts.resolution;
        let size = opts.size;
        let mask_data = opts.mask_data;
        let data = opts.data;
        let (rtree, bounds, particles) = initialize_data(data, &mask_data, SimpleBounds::from_extent(extent), resolution, size, Some(particle_count as usize));
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
            size,
            depth: 0,
            mask_data
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
            particle.status = 0;
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

    // 파티클을 추가하는 메서드
    fn add_particles(&mut self, num_to_add: usize) {
        let borrow_ref = self.tiling_system.borrow();
        if let Some(root_tiles) = borrow_ref.tiles_by_depth.get(&0) {
            let &unwrap_root_tile = &root_tiles.get(0).unwrap();
            let rtree = &unwrap_root_tile.rtree;
            let rtree_ref = rtree;
            let bounds = rtree_ref.root().envelope();
            let simple_bounds = SimpleBounds::from_aabb_point(&bounds);

            let particles = generate_particles(SimpleBounds::from_extent(self.extent), num_to_add, &self.mask_data, simple_bounds, self.resolution, self.size);
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

    pub fn drawCanvasCurrentFlow(&mut self, options:JsValue) -> Result<HtmlCanvasElement, JsValue> {
        let window = window().ok_or("Could not obtain window")?;
        let document = window.document().ok_or("Could not obtain document")?;

        let canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;

        if options.is_null() || options.is_undefined() {
            log!("error: {:?}", options);
        } else {
            //log!("draw canvas start input options: {:?}", options);
            let opts: DrawCurrentFlowCanvasOption = serde_wasm_bindgen::from_value(options)?;
            let size = self.size;
            let raw_extent = self.extent;
            let extent = SimpleBounds::from_extent(raw_extent);
            let resolution = self.resolution;
            let line_width = opts.line_width;
            let life = opts.life;
            let exaggeration = opts.exaggeration;

            canvas.set_height(size[1] as u32);
            canvas.set_width(size[0] as u32);

            let ctx = canvas.get_context("2d")?.unwrap().dyn_into::<CanvasRenderingContext2d>()?;
            ctx.clear_rect(0.0, 0.0, size[0], size[1]);

            let borrow_ref = self.tiling_system.borrow();
            if let Some(root_tiles) = borrow_ref.tiles_by_depth.get(&0) {
                let &unwrap_root_tile = &root_tiles.get(0).unwrap();
                let rtree = &unwrap_root_tile.rtree;

                for (_, particle) in self.particles.iter_mut().enumerate() {
                    let pixel = to_pixel(particle.coordinate.longitude, particle.coordinate.latitude, extent, resolution);
                    let rtree_ref = rtree;
                    let bounds = rtree_ref.root().envelope();
                    let simple_bounds = SimpleBounds::from_aabb_point(&bounds);

                    handle_particle(particle, rtree_ref, &self.interpolation_type, &exaggeration, pixel, &self.mask_data, extent, resolution, size, life);

                    if pixel[0] < 0.0 || pixel[0] > size[0] || pixel[1] < 0.0 || pixel[1] > size[1] || particle.loop_count >= life {
                        *particle = generate_particle(extent, &self.mask_data, simple_bounds, self.resolution, self.size);
                    }

                    draw_particle(&ctx, particle, extent, resolution, pixel, line_width);
                }
            }
        }
        Ok(canvas)

    }
}

fn handle_particle(
    particle: &mut Particle,
    rtree: &RTree<WeatherData>,
    interpolation_type: &String,
    exaggeration: &f64,
    pixel: [f64; 2],
    mask_data: &Vec<u8>,
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
        *particle = generate_particle(extent, mask_data, data_bounds, resolution, size);
        return;
    }

    let new_longitude = particle.coordinate.longitude + interpolated_u * exaggeration;
    let new_latitude = particle.coordinate.latitude + interpolated_v * exaggeration;
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
}

// 파티클 상태 업데이트 함수
fn update_particle_state(
    particle: &mut Particle,
    mask_data: &Vec<u8>,
    data_bounds: SimpleBounds,
    extent: SimpleBounds,
    resolution: f64,
    size: [f64; 2],
    life: i16
) {
    if particle.prev_coordinats.len() >= 20 {
        particle.prev_coordinats.remove(0);
    }
    particle.prev_coordinats.push(particle.coordinate);
    particle.loop_count += 1;

    // 파티클이 수명이 다한 경우 다시 생성
    if should_respawn(particle, life) {
        *particle = generate_particle(extent, mask_data, data_bounds, resolution, size);
    }
}

fn interpolate_uv(
    particle: &mut Particle,
    rtree: &RTree<WeatherData>,
    interpolation_type:&String,
    mask_data: &Vec<u8>,
    extent: SimpleBounds,
    resolution: f64,
    size: [f64; 2],
) -> (f64, f64) {
    let i_type = interpolation_type.as_str();
    let rtree_ref = rtree;

    let (u, v) = match i_type {
        "IDW" => interpolate_by_inverse_distance_weighted(particle.coordinate, rtree_ref, Some(16), mask_data, extent, resolution, size),
        "KRIGING" => interpolate_by_kriging(particle.coordinate, rtree_ref, Some(6), mask_data, extent, resolution, size),
        /*"BILINEAR" => interpolate_by_bilinear(particle.coordinate, rtree_ref, mask_data, extent, resolution, size),*/
        "BICUBIC" => interpolate_by_bicubic(particle.coordinate, rtree_ref, mask_data, extent, resolution, size),
        _ => (0.0, 0.0),  // 기본값 처리
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
        particle.status = 3;
    } else {
        particle.current_u = detour_u;
        particle.current_v = detour_v;

        particle.avoid_interpolation_frame_count = Some(1);

        particle.status = 2;
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

    particle.status = 1;
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
            particle.status = 3;
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

fn should_respawn(particle: &Particle, life: i16) -> bool {
    (particle.current_u.abs() < 0.01 && particle.current_v.abs() < 0.01) || particle.loop_count >= life  || particle.status == 3
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

    ctx.save();

    ctx.translate(pixel[0], pixel[1]).expect("ctx translate failed");
    ctx.rotate(- (std::f64::consts::PI / 2.0) + rotate).expect("ctx rotate failed");
    draw_arrow(&ctx, speed).expect("draw arrow failed");

    ctx.restore();

    //draw_debug_vector(ctx, point.udata, point.vdata, lon, lat, pixel[0], pixel[1]);
}

fn draw_arrow(ctx: &CanvasRenderingContext2d, speed:f64,) -> Result<(), JsValue> {
    let dpi =  96.0; // 일반적으로 96dpi로 계산
    let pixels_per_cm = dpi / 2.54; // 1cm에 해당하는 픽셀 수
    let arrow_length = (speed  * pixels_per_cm).min(pixels_per_cm * 1.5);

    let arrow_head_size = 4.0/*(5.0 / speed).clamp(1.5, 10.0)*/;   // 머리 고정 크기
    let offset = arrow_length / 2.0 - 2.0;
    let half_width = 3.0;  // 몸통 고정 너비
    let arrow_tail_length =  (arrow_length - arrow_head_size).max(0.0);

    let width = half_width * 2.0;
    let curve_radius = 1.25;
    let adjusted_tail_length = arrow_tail_length - curve_radius;

    let gradient = ctx.create_linear_gradient(0.0, 0.0, -adjusted_tail_length, 0.0);
    gradient.add_color_stop(0.0, &*calculate_color_from_speed(speed, 1.0));
    gradient.add_color_stop(1.0, &*calculate_color_from_speed(speed, calculate_alpha_from_speed(speed)));

    ctx.set_fill_style(&gradient.into());
    ctx.set_stroke_style(&JsValue::from_str("#EAEAEA"));
    ctx.set_line_width(0.6);

    // 화살표 꼬리 (직사각형)
    ctx.begin_path();
    ctx.move_to(0.0 + offset, -width*(1.2 / speed).clamp(1.2, 1.8)); // 머리 좌측
    ctx.line_to(arrow_head_size + offset, 0.0); // 머리 끝점 (삼각형 끝, 가운데)
    ctx.line_to(0.0 + offset, width*(1.2 / speed).clamp(1.2, 1.8)); // 머리 우측
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

fn initialize_data(data_array: Vec<(f64,f64,f64,f64)>, mask_data:&Vec<u8>, extent: SimpleBounds, resolution: f64, size:[f64;2], particle_count: Option<usize>) -> (RTree<WeatherData>, SimpleBounds, Option<Vec<Particle>>) {
    let mut rtree = RTree::new();
    //log!("데이터 추출 전");
    let _ = for data in data_array.iter() {
        //log!("data {:?}", data);
        // 각 item이 객체라고 가정하고 필요한 필드를 추출
        let longitude = data.0; //Reflect::get(&data, &JsValue::from_str("longitude")).ok().unwrap().as_f64().unwrap();

        let latitude = data.1;// Reflect::get(&data, &JsValue::from_str("latitude")).ok().unwrap().as_f64().unwrap();
        let udata = data.2;// Reflect::get(&data, &JsValue::from_str("udata")).ok().unwrap().as_f64().unwrap();
        let vdata = data.3;// Reflect::get(&data, &JsValue::from_str("vdata")).ok().unwrap().as_f64().unwrap();

        //log!("longitude {:?}, latitude {:?}, udata {:?}, vdata {:?}", longitude, latitude, udata, vdata);
        let coord = wgs84_to_web_mercator(longitude, latitude);
        let pixel = to_pixel(coord.longitude, coord.latitude, extent, resolution);

        //log!("coord {:?}, pixel {:?}", coord, pixel);
        if !is_masked(pixel, mask_data, size[0], size[1]) {
            let wd = WeatherData {
                coordinate: coord,
                udata,
                vdata,
            };
            rtree.insert(wd);
        }
    };
    //log!("데이터 추출 종료");
    let bounds = rtree.root().envelope();

    let simple_bounds = SimpleBounds::from_aabb_point(&bounds);
    //log!("바운드 생성");
    let particles = if let Some(particle_count) = particle_count {
        Some(generate_particles(extent, particle_count, mask_data, simple_bounds, resolution, size))
    } else {
        None
    };

    (rtree, simple_bounds, particles)
}

fn generate_particles(extent:SimpleBounds,num_particles: usize,mask_data:&Vec<u8>, data_bounds:SimpleBounds, resolution: f64, size:[f64;2]) -> Vec<Particle> {
    let mut particles = Vec::new();
    for _ in 0..num_particles {
        particles.push(generate_particle(extent,mask_data, data_bounds, resolution, size));
    }

    particles
}

fn generate_particle(extent:SimpleBounds, mask_data:&Vec<u8>, data_bounds:SimpleBounds, resolution: f64, size:[f64;2]) -> Particle {
    loop {
        let width = size[0];
        let height = size[1];

        let (x, y) = generate_random_pixel(width as usize, height as usize);
        let [longitude, latitude] = to_coord([x as f64, y as f64], extent, resolution);
        let coordinate = Coordinate { longitude, latitude};

        if !is_masked([x as f64, y as f64], mask_data, width, height) && data_bounds.containsCoordinate(coordinate) {
            return Particle {
                coordinate,
                original_coordinate: coordinate,
                prev_coordinats: Vec::new(),
                current_u: 0.0,
                current_v: 0.0,
                loop_count: 0,
                avoid_interpolation_frame_count: None,
                status: 0
            };
        }
    }
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
    sample_size: usize
) -> HashMap<usize, Vec<WeatherData>> {
    let clusters: HashMap<usize, Vec<WeatherData>> = cluster_labels_to_clusters(points, cluster_labels);

    // 클러스터별로 랜덤 샘플링
    let mut sampled_points: HashMap<usize, Vec<WeatherData>> = HashMap::new();
    for (cluster_id, cluster_points) in clusters.iter() {
        let sample_size = sample_size.min(cluster_points.len()); // 클러스터의 포인트보다 큰 샘플은 방지
        let random_indices = get_random_index(cluster_points.len(), sample_size); // 랜덤 인덱스 생성

        let sample: Vec<WeatherData> = random_indices
            .iter()
            .map(|&i| cluster_points[i].clone()) // 인덱스를 기반으로 포인트 복사
            .collect();

        sampled_points.insert(*cluster_id, sample);
    }

    sampled_points
}

// 랜덤 인덱스를 생성하는 함수
pub fn get_random_index(max: usize, count: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut buffer = vec![0u8; count]; // count만큼의 난수 생성 버퍼

    getrandom(&mut buffer).expect("Failed to generate random bytes");

    for &byte in buffer.iter() {
        // 생성된 난수를 max 범위 안에 맞게 변환
        let index = (byte as usize) % max;
        indices.push(index);
    }

    indices
}
