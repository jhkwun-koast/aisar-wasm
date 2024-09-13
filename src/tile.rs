use rstar::{RTree, AABB};
use serde::Serialize;
use crate::{ SimpleBounds};
use crate::coordinate::WeatherData;

macro_rules! log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()));
}
// Tile 구조체 (각 타일은 R-tree를 가짐)
#[derive(Clone, Serialize, Debug)]
pub struct Tile {
    #[serde(skip)]
    pub rtree: RTree<WeatherData>,
    pub bounds: SimpleBounds,
    pub depth: i32,
}

impl Tile {
    pub fn new(rtree: RTree<WeatherData>, bounds: SimpleBounds, depth: i32) -> Self {
        Tile { rtree, bounds, depth }
    }

    pub fn get_points(&self) -> Vec<WeatherData> {
        self.rtree.iter().cloned().collect()
    }

    // 타일 내 포인트 갯수 반환
    pub fn point_count(&self) -> usize {
        self.rtree.size()
    }

    // 타일에 점을 추가
    pub fn insert_point(&mut self, point: WeatherData) {
        self.rtree.insert(point);
    }

    // 타일 내에서 특정 범위의 포인트 검색
    pub fn search_points(&self, bounds: AABB<WeatherData>) -> Vec<WeatherData> {
        self.rtree.locate_in_envelope(&bounds).copied().collect::<Vec<WeatherData>>()
    }

    // 타일 내에서 가장 가까운 두 점 찾기
    pub fn find_nearest_points(&self, coords: WeatherData, k: usize) -> Vec<WeatherData> {
        self.rtree.nearest_neighbor_iter(&coords).take(k).copied().collect::<Vec<WeatherData>>()
    }
    // RTree를 삭제하는 메서드
    pub fn clear_rtree(&mut self) {
        // RTree의 내용을 비운다. (RTree가 벡터 또는 다른 컨테이너일 경우)
        self.rtree = RTree::new();
    }
    // 타일을 4개로 분할하는 함수
    pub fn divide(&self) -> Vec<Tile> {
        let (min, max) = (self.bounds.lower(), self.bounds.upper());

        let [_, min_y] = min;
        let [max_x, max_y] = max;
        let mid_x = (min[0] + max[0]) / 2.0;
        let mid_y = (min[1] + max[1]) / 2.0;


        let min_point = min;
        let mid_point = [mid_x, mid_y];
        let center_bottom_point = [mid_x, min_y];
        let center_right_point = [max_x, mid_y];
        let center_left_point = [max_x, mid_y];
        let center_top_point = [mid_x, max_y];
        let max_point = [max_x, max_y];

        let quadrants = vec![
            SimpleBounds::from_corners(min_point, mid_point),
            SimpleBounds::from_corners(center_bottom_point, center_right_point),
            SimpleBounds::from_corners(center_left_point, center_top_point),
            SimpleBounds::from_corners(min_point, max_point),
        ];
        let new_depth = self.depth + 1;

        // 각 사분면에 대해 서브타일 생성
        quadrants
            .into_iter()
            .map(|bounds| Tile::new(RTree::new(), bounds, new_depth))  // 새 타일 생성
            .collect()
    }
}