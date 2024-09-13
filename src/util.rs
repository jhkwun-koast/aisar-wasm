use crate::SimpleBounds;
use getrandom::getrandom;

macro_rules! log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()));
}

pub fn to_pixel (x:f64, y:f64, extent:SimpleBounds, resolution:f64) -> [f64; 2] {
    let size = calculate_size(extent, resolution);
    let x1:f64 = (x - extent.min_x) / (extent.max_x - extent.min_x) * size[0];
    let y1:f64 = (extent.max_y - y) / (extent.max_y - extent.min_y) * size[1];

    /*let a = web_mercator_to_wgs84(x, y);

    log!("size {:?}, x {:?}, y {:?}, x1 {:?}, y1 {:?}, x2 {:?}, y2 {:?}", size, x, y, x1, y1, a.longitude, a.latitude);*/
    return [x1, y1];
}

pub fn to_coord(pixel: [f64; 2], extent: SimpleBounds, resolution: f64) -> [f64; 2] {
    let size = calculate_size(extent, resolution);
    let [x, y] = pixel;
    let [width, height] = size;
    let min_x = extent.min_x;
    let max_x = extent.max_x;
    let min_y = extent.min_y;
    let max_y = extent.max_y;

    [min_x + ((max_x- min_x) * x / width),  min_y + ((max_y - min_y) * (height - y) /height)]
}

fn calculate_size(extent:SimpleBounds, resolution:f64) -> [f64; 2] {
    let width = extent.width() / resolution;
    let height = extent.height() / resolution;
    return [width, height];
}

pub fn generate_random_f64(min: f64, max: f64) -> f64 {
    let mut buf = [0u8; 8];  // 64비트 (8바이트) 배열 생성
    getrandom(&mut buf).expect("Failed to get random bytes");
    let random_value = u64::from_le_bytes(buf) as f64 / u64::MAX as f64;  // 0.0 ~ 1.0 범위
    min + (random_value * (max - min))
}


pub fn generate_random_pixel(width: usize, height: usize) -> (usize, usize) {
    let x = generate_random_usize(0, width);
    let y = generate_random_usize(0, height);
    (x, y)
}

// 범위 내에서 랜덤한 usize 값 생성
pub fn generate_random_usize(min: usize, max: usize) -> usize {
    let mut buf = [0u8; 8];
    getrandom(&mut buf).expect("Failed to get random bytes");
    let random_value = u64::from_le_bytes(buf) as f64 / u64::MAX as f64;
    min + (random_value * (max - min) as f64) as usize
}

// RGBA 형식의 인덱스 계산 유틸리티 함수
pub fn calculate_pixel_index(x: usize, y: usize, width: usize, height: usize) -> Option<usize> {
    if x < width && y < height {
        Some((y * width + x) * 4)  // RGBA는 4바이트씩 차지하므로 * 4
    } else {
        None  // 유효하지 않은 좌표라면 None 반환
    }
}