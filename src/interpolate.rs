use rstar::RTree;
use crate::coordinate::{Coordinate, WeatherData};
use crate::{is_masked, SimpleBounds};
use crate::util::to_pixel;

fn find_min_data_in_bounds<'a> (
    coordinate: Coordinate,
    rtree: &'a RTree<WeatherData>,
    min_points:Option<usize>,
    buffer_size:Option<f64>,
    ratio:Option<f64>,
    mask_data: &'a Vec<u8>,
    extent: SimpleBounds,
    resolution: f64,
    size: [f64;2]
) -> Vec<&'a WeatherData> {
    let min_points = min_points.unwrap_or(4);
    let mut buffer_size = buffer_size.unwrap_or(10000.0);
    let ratio = ratio.unwrap_or( 1.5);

    let max_attempts = 4; // 최대 시도 횟수
    let mut attempts = 0; // 현재 시도 횟수

    let mut bounds = coordinate.to_bounds(buffer_size);
    let mut points = Vec::new();

    while points.len() < min_points && attempts < max_attempts  {
        // SimpleBounds를 AABB로 변환
        let bounds_aabb = bounds.to_aabb(); // SimpleBounds를 AABB로 변환하는 함수가 있다고 가정

        // R-tree에서 포인트 추출
        points = rtree.locate_in_envelope(&bounds_aabb)
            .filter(|weather_data| {
                let pixel = to_pixel(weather_data.coordinate.longitude, weather_data.coordinate.latitude, extent, resolution);
                !is_masked(pixel, mask_data, size[0], size[1])  // 육지 여부 확인
            })
            .collect();

        // 3. 최소 포인트에 도달하지 않았으면 버퍼 확장
        buffer_size *= ratio;  // 버퍼 확장
        bounds = coordinate.to_bounds(buffer_size);
        attempts += 1;
    }

    points
}

pub fn interpolate_by_kriging(particle_coord: Coordinate, rtree: &RTree<WeatherData>, min_points: Option<usize>, mask_data: &Vec<u8>, extent: SimpleBounds, resolution: f64, size: [f64; 2]) -> (f64, f64) {
    let points = find_min_data_in_bounds(particle_coord, rtree, min_points, None, None,mask_data,extent,resolution,size);

    if points.len() == 0 {
        return (0.0, 0.0);
    }

    let distances: Vec<f64> = points.iter()
        .map(|data| data.coordinate.distance_to(&particle_coord).abs())
        .collect();
    let weights = calculate_kriging_weights(&distances, &|d| spherical_variogram(d));

    // 크리깅 보간법을 사용하여 계산
    let mut u_total = 0.0;
    let mut v_total = 0.0;

    for (i, data) in points.iter().enumerate() {
        u_total += weights[i] * data.udata;
        v_total += weights[i] * data.vdata;
    }

    (u_total, v_total)
}

fn spherical_variogram(distance: f64) -> f64 {
    let range = 1200.0;
    let sill = 1.5;

    if distance < range {
        sill * (1.5 * (distance / range) - 0.5 * (distance / range).powi(3))
    } else {
        sill
    }
}

// 반변동도 기반 크리깅 가중치 계산
pub fn calculate_kriging_weights(
    distances: &Vec<f64>,
    variogram: &impl Fn(f64) -> f64
) -> Vec<f64> {
    let n = distances.len();
    let mut weights = vec![0.0; n];

    for i in 0..n {
        let mut weight_sum = 0.0;
        for j in 0..n {
            weight_sum += variogram(distances[i] - distances[j]);
        }
        weights[i] = 1.0 / weight_sum;  // 반변동도에 따라 가중치 계산
    }

    weights
}

pub fn interpolate_by_inverse_distance_weighted (particle_coord:Coordinate, rtree: &RTree<WeatherData>, min_points: Option<usize>, mask_data: &Vec<u8>, extent: SimpleBounds, resolution: f64, size: [f64; 2]) -> (f64, f64) {
    let points = find_min_data_in_bounds(particle_coord, rtree, min_points, None, None,mask_data,extent,resolution,size);

    if points.len() == 0 {
        return (0.0, 0.0);
    }

    let mut weighted_sum_u = 0.0;
    let mut weighted_sum_v = 0.0;
    let mut weight_total = 0.0;

    for weather_data in points {

        let dist = ((weather_data.coordinate.longitude - particle_coord.longitude).powi(2) + (weather_data.coordinate.latitude - particle_coord.latitude).powi(2)).sqrt();
        let dist = dist.max(2000.0);

        let direction_similarity = calculate_direction_similarity(
            particle_coord,
            (weather_data.udata, weather_data.vdata)
        );

        let weight = (1.0 / dist.powf(2.0)) * direction_similarity;

        // U, V 값에 가중치 적용
        weighted_sum_u += weight * weather_data.udata;
        weighted_sum_v += weight * weather_data.vdata;
        weight_total += weight;
    }

    if weight_total > 0.0 {
        (weighted_sum_u / weight_total, weighted_sum_v / weight_total)
    } else {
        (0.0, 0.0)  // 기본값 반환 (가중치 합이 0인 경우)
    }
}

pub fn calculate_direction_similarity(
    particle_coord: Coordinate,
    data_uv: (f64, f64)  // (udata, vdata)
) -> f64 {
    let particle_u = particle_coord.longitude;
    let particle_v = particle_coord.latitude;

    // 내적을 통한 유사도 계산
    let dot_product = particle_u * data_uv.0 + particle_v * data_uv.1;
    let magnitude_particle = (particle_u.powi(2) + particle_v.powi(2)).sqrt();
    let magnitude_data = (data_uv.0.powi(2) + data_uv.1.powi(2)).sqrt();

    // 유사도 = 벡터 내적 / 크기 곱
    let similarity = if magnitude_particle > 0.0 && magnitude_data > 0.0 {
        dot_product / (magnitude_particle * magnitude_data)
    } else {
        0.0
    };

    similarity.max(0.0)  // 유사도가 음수일 경우 0으로 처리
}

pub fn interpolate_by_bilinear(particle_coord: Coordinate, rtree: &RTree<WeatherData>, mask_data: &Vec<u8>, extent: SimpleBounds, resolution: f64, size: [f64; 2]) -> (f64, f64) {
    let points = find_min_data_in_bounds(particle_coord, rtree, Some(4), None, None,mask_data,extent,resolution,size);

    // 4개의 주변 포인트 찾기
    if points.len() < 4 {
        return (0.0, 0.0); // 충분한 포인트가 없으면 기본값 반환
    }

    // Bilinear interpolation 계산
    let (mut u_sum, mut v_sum, mut total_weight) = (0.0, 0.0, 0.0);

    for weather_data in points.iter() {
        let weight = 1.0 / particle_coord.distance_to(&weather_data.coordinate); // 선형 가중치
        u_sum += weight * weather_data.udata;
        v_sum += weight * weather_data.vdata;
        total_weight += weight;
    }

    (u_sum / total_weight, v_sum / total_weight)
}

pub fn interpolate_by_bicubic(particle_coord: Coordinate, rtree: &RTree<WeatherData>, mask_data: &Vec<u8>, extent: SimpleBounds, resolution: f64, size: [f64; 2]) -> (f64, f64) {
    let points = find_min_data_in_bounds(particle_coord, rtree, Some(16), None, None,mask_data,extent,resolution,size);

    // 16개의 주변 포인트 찾기
    if points.len() < 16 {
        return (0.0, 0.0); // 충분한 포인트가 없으면 기본값 반환
    }

    // 거리를 캐싱하여 나중에 사용
    let mut cached_distances: Vec<f64> = Vec::with_capacity(points.len());

    for weather_data in points.iter() {
        let distance = particle_coord.distance_to(&weather_data.coordinate);
        cached_distances.push(distance); // 거리 저장
    }

    // Bicubic interpolation 계산
    let (mut u_sum, mut v_sum, mut total_weight) = (0.0, 0.0, 0.0);

    for (i, weather_data) in points.iter().enumerate() {
        let distance = cached_distances[i];  // 저장된 거리값 사용
        if distance > 0.0 {
            let weight = 1.0 / distance.powi(3); // 3차 가중치
            u_sum += weight * weather_data.udata;
            v_sum += weight * weather_data.vdata;
            total_weight += weight;
        }
    }

    (u_sum / total_weight, v_sum / total_weight)
}