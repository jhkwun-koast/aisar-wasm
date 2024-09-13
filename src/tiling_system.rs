use std::collections::HashMap;
use serde::{Serialize};
use web_sys::{console};
use crate::coordinate::WeatherData;
use crate::tile::Tile;

macro_rules! log {
    ($($t:tt)*) => (console::log_1(&format!($($t)*).into()));
}

#[derive(Serialize)]
pub struct TilingSystem {
    pub(crate) tiles_by_depth: HashMap<i32, Vec<Tile>>, // 깊이별로 타일을 분리 저장
    max_depth: i32,
}

impl TilingSystem {
    pub fn new(max_depth: i32) -> Self {
        Self {
            tiles_by_depth: HashMap::new(),
            max_depth,
        }
    }

    // 타일을 깊이별로 저장하는 함수
    fn add_tile(&mut self, tile: Tile) {
        let depth = tile.depth;
        self.tiles_by_depth
            .entry(depth)
            .or_insert_with(Vec::new)
            .push(tile);
    }
    // 타일을 미리 생성하는 함수
    pub fn generate_tiles(&mut self) {
        let mut tiles_to_generate: Vec<Tile> = Vec::new();

        // 이미 생성된 0레벨 타일을 이용하여 리스트에 추가
        if let Some(root_tiles) = self.tiles_by_depth.get(&0) {
            tiles_to_generate.extend(root_tiles.clone()); // 0 레벨의 타일들을 가져옴
        }

        // max_depth까지 타일을 생성
        for current_depth in 1..=self.max_depth {
            let mut new_tiles = Vec::new();

            // 현재 깊이의 타일들을 순회하며 분할
            for tile in tiles_to_generate {
                // 타일을 분할하여 새로운 서브타일 추가
                let sub_tiles = tile.divide(); // divide는 새로운 서브타일을 반환
                new_tiles.extend(sub_tiles);
            }

            // 새로 생성된 타일들을 다음 깊이에 추가
            self.tiles_by_depth
                .entry(current_depth)
                .or_insert_with(Vec::new)
                .extend(new_tiles.clone());

            // 다음 깊이를 위해 new_tiles에 저장된 타일들로 업데이트
            tiles_to_generate = new_tiles;
        }
    }
    // 부모 타일의 R-tree에 있는 포인트를 해당 타일로 분배하는 함수
    pub fn distribute_points_to_tiles(&mut self) {
        // 1. 부모 타일을 가져옴 (깊이 0에 해당하는 타일들)
        if let Some(root_tiles) = self.tiles_by_depth.get(&0) {
            let mut points_to_distribute = vec![];

            // 2. 첫 번째 루프: 불변 참조로 부모 타일의 포인트를 미리 수집
            for parent_tile in root_tiles.iter() {
                let points: Vec<WeatherData> = parent_tile.rtree.iter().cloned().collect();
                points_to_distribute.extend(points);
            }

            // 3. 두 번째 루프: 가변 참조로 각 포인트를 해당하는 타일에 삽입
            for point in points_to_distribute {
                if let Some(deepest_tile_key) = self.find_tile_key_for_point(&point) {
                    // 타일 키를 바탕으로 가변 참조로 타일에 접근하여 포인트 삽입
                    if let Some(target_tile) = self.tiles_by_depth.get_mut(&deepest_tile_key.0) {
                        if let Some(tile) = target_tile.iter_mut().find(|t| t.bounds.contains(&point)) {
                            tile.insert_point(point);
                        }
                    }
                }
            }
        } else {
            log!("No root tiles found at depth 0");
        }
    }

    // 포인트가 속하는 타일의 키를 찾아주는 함수 (불변 참조로만 사용)
    pub(crate) fn find_tile_key_for_point(&self, point: &WeatherData) -> Option<(i32, usize)> {
        let mut found_tile_key: Option<(i32, usize)> = None;

        // 가장 깊은 타일까지 탐색
        for depth in 1..=self.max_depth {
            if let Some(tiles_at_depth) = self.tiles_by_depth.get(&depth) {
                for (i, tile) in tiles_at_depth.iter().enumerate() {
                    if tile.bounds.contains(point) {
                        // 포인트가 포함된 타일을 찾았을 때도, 계속해서 더 깊은 타일이 있는지 탐색
                        found_tile_key = Some((depth, i));
                    }
                }
            }
        }

        found_tile_key // 가장 깊은 타일의 키 반환
    }
}
