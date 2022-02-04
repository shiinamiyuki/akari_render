use std::ops::Index;

use crate::{util::arrayvec::VirtualStorage, RobustSum};

use super::arrayvec::{ArrayVec, DynStorage};

pub struct TiledArray2D<T, const TILE_SIZE: usize> {
    data: ArrayVec<T, DynStorage<T>>,
    #[allow(dead_code)]
    size: [usize; 2],
    tiles: [usize; 2],
    average: T,
}

impl<T, const TILE_SIZE: usize> TiledArray2D<T, TILE_SIZE>
where
    T: 'static
        + Clone
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<f32, Output = T>,
{
    fn compute_index(&self, x: usize, y: usize) -> usize {
        let tx = x / TILE_SIZE;
        let ty = y / TILE_SIZE;
        let ox = x % TILE_SIZE;
        let oy = y % TILE_SIZE;
        (tx + ty * self.tiles[0]) * TILE_SIZE * TILE_SIZE + ox + oy * TILE_SIZE
    }
    pub fn average(&self)->T{
        self.average
    }
    pub fn new<F: Fn(usize, usize) -> T>(
        size: [usize; 2],
        f: F,
        default: T,
        virtual_memory: bool,
    ) -> Self {
        let tiles = [
            (size[0] + TILE_SIZE - 1) / TILE_SIZE,
            (size[1] + TILE_SIZE - 1) / TILE_SIZE,
        ];
        let len = tiles[0] * tiles[1] * TILE_SIZE * TILE_SIZE;
        let mut data = unsafe {
            if !virtual_memory {
                ArrayVec::<T, DynStorage<T>>::from_storage(
                    Box::new(Vec::with_capacity(len)) as DynStorage<_>
                )
            } else {
                ArrayVec::<T, DynStorage<T>>::from_storage(
                    Box::new(VirtualStorage::new(len)) as DynStorage<_>
                )
            }
        };
        for _ in 0..data.capacity() {
            data.push(default).unwrap();
        }
        let mut a = Self {
            size,
            tiles,
            data,
            average: default,
        };
        let mut sum = None;
        for y in 0..size[1] {
            for x in 0..size[0] {
                let i = a.compute_index(x, y);
                let v = f(x, y);
                a.data[i] = v;
                if let None = sum {
                    sum = Some(RobustSum::new(v));
                } else {
                    sum.as_mut().unwrap().add(v);
                }
            }
        }
        a.average = sum.unwrap().sum() / (size[0] * size[1]) as f32;
        a
    }
}
impl<T, const N: usize> Index<(usize, usize)> for TiledArray2D<T, N>
where
    T: 'static
        + Clone
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<f32, Output = T>,
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let i = self.compute_index(index.0, index.1);
        &self.data[i]
    }
}
