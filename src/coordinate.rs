use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use crate::SimpleBounds;
use rstar::{Point as  RstarPoint};

#[derive(Debug, Clone, Copy,Serialize, Deserialize)]
pub struct Coordinate {
    pub longitude: f64,
    pub latitude: f64,
}
impl PartialEq for Coordinate {
    fn eq(&self, other: &Self) -> bool {
        Coordinate::eq(&self, &other)
    }
}

impl RstarPoint for Coordinate {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Coordinate {
            longitude: generator(0),
            latitude: generator(1),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.longitude,  // 0번째 인덱스는 경도(longitude)
            1 => self.latitude,   // 1번째 인덱스는 위도(latitude)
            _ => panic!("Index out of bounds"), // 2 이상의 인덱스는 에러
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.longitude,  // 0번째 인덱스는 경도(longitude)
            1 => &mut self.latitude,   // 1번째 인덱스는 위도(latitude)
            _ => panic!("Index out of bounds"),   // 2 이상의 인덱스는 에러
        }
    }
}

impl Coordinate {
    pub fn new(longitude: f64, latitude: f64) -> Self {
        Coordinate { longitude, latitude }
    }

    pub fn eq(&self, other: &Self) -> bool {
        let offset = 0.000000001;
        let lon_delta = (self.longitude - other.longitude).abs();
        let lat_delta = (self.latitude - other.latitude).abs();

        lon_delta < offset && lat_delta < offset
    }

    pub fn from_array(arr: [f64; 2]) -> Self {
        let [longitude, latitude] = arr;
        Coordinate { longitude, latitude }
    }

    pub fn distance_to (&self, other: &Coordinate) -> f64 {
        ((self.longitude - other.longitude).powi(2) + (self.latitude - other.latitude).powi(2)).sqrt()
    }

    pub fn to_bounds (&self, buffer_size:f64) -> SimpleBounds {
        SimpleBounds {
            min_x: self.longitude - buffer_size,
            max_x: self.longitude + buffer_size,
            min_y: self.latitude - buffer_size,
            max_y: self.latitude + buffer_size,
        }
    }
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WeatherData {
    pub(crate) coordinate: Coordinate,
    pub(crate) udata: f64,
    pub(crate) vdata: f64,
}

impl WeatherData {
    pub fn speed(&self) -> f64 {
        (self.udata.powi(2) + self.vdata.powi(2)).sqrt()
    }

    pub fn rotate(&self) -> f64 {
        self.udata.atan2(self.vdata)
    }

    pub fn direction_difference(&self, point2: &WeatherData) -> f64 {
        let dot_product = self.udata * point2.udata + self.vdata * point2.vdata;
        let magnitude1 = self.speed();
        let magnitude2 = point2.speed();
        let cos_theta = dot_product / (magnitude1 * magnitude2);
        cos_theta.acos() // 각도 차이를 라디안으로 반환
    }
}

impl PartialEq for WeatherData {
    fn eq(&self, other: &Self) -> bool {
        Coordinate::eq(&self.coordinate, &other.coordinate) && self.udata == other.udata && self.vdata == other.vdata
    }
}

// rstar::Point 트레이트 구현

impl RstarPoint for WeatherData {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {

        WeatherData {
            coordinate: Coordinate::new(generator(0), generator(1)),
            udata: 0.0,
            vdata: 0.0,
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.coordinate.longitude,  // 0번째 인덱스는 경도(longitude)
            1 => self.coordinate.latitude,   // 1번째 인덱스는 위도(latitude)
            _ => panic!("Index out of bounds"), // 2 이상의 인덱스는 에러
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.coordinate.longitude,  // 0번째 인덱스는 경도(longitude)
            1 => &mut self.coordinate.latitude,   // 1번째 인덱스는 위도(latitude)
            _ => panic!("Index out of bounds"),   // 2 이상의 인덱스는 에러
        }
    }
}


// EPSG:4326 -> EPSG:3857 변환
pub fn wgs84_to_web_mercator(lon: f64, lat: f64) -> Coordinate {
    let x = lon * 20037508.34 / 180.0;
    let y = ((PI / 4.0) + (lat.to_radians() / 2.0)).tan().ln() * 20037508.34 / PI;
    Coordinate::new(x,y)
}

// EPSG:3857 -> EPSG:4326 변환
pub fn web_mercator_to_wgs84(x: f64, y: f64) -> Coordinate {
    let lon = x * 180.0 / 20037508.34;
    let lat = (2.0 * ((y * PI / 20037508.34).exp()).atan() - PI / 2.0).to_degrees();
    Coordinate::new(lon,lat)
}