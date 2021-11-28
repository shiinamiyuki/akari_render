use crate::*;
use glm::IVec2;
use rand::Rng;

use crate::sobolmat::SOBOL_MATRIX;
pub trait Sampler: Sync + Send {
    // fn start_pixel(&mut self, px: &IVec2, res: &IVec2);
    fn start_next_sample(&mut self);
    fn next1d(&mut self) -> Float;
    fn next2d(&mut self) -> Vec2 {
        vec2(self.next1d(), self.next1d())
    }
    fn next3d(&mut self) -> Vec3 {
        vec3(self.next1d(), self.next1d(), self.next1d())
    }
}

pub struct PCG {
    state: usize,
}
impl PCG {
    const MULTIPILER: usize = 6364136223846793005;
    const INC: usize = 1442695040888963407;

    pub fn pcg32(&mut self) -> u32 {
        let mut x = self.state;
        let count = x >> 59;
        self.state = x.wrapping_mul(Self::MULTIPILER).wrapping_add(Self::INC); //x * Self::MULTIPILER + Self::INC;
        x ^= x >> 18;
        ((x >> 27) as u32).rotate_right(count as u32)
    }
    pub fn new(seed: usize) -> Self {
        let mut r = Self {
            state: seed.wrapping_add(Self::INC),
        };
        let _ = r.pcg32();
        r
    }
}
pub struct PCGSampler {
    pub rng: PCG,
}
impl Sampler for PCGSampler {
    fn next1d(&mut self) -> Float {
        self.rng.pcg32() as Float / (std::u32::MAX as Float)
    }
    // fn start_pixel(&mut self, px: &IVec2, res: &IVec2) {}
    fn start_next_sample(&mut self) {}
}

pub struct SobolSampler {
    dim: u32,
    rotation: u32,
    index: u32,
}
fn cmj_hash_simple(mut i: u32, p: u32) -> u32 {
    i = (i ^ 61) ^ p;
    // i += i << 3;
    i = i.wrapping_add(i << 3);
    i ^= i >> 4;
    // i *= 0x27d4eb2d;
    i = i.wrapping_mul(0x27d4eb2d);
    i
}
fn sobol(dim: u32, mut i: u32, rng: u32) -> f32 {
    let mut res: u32 = 0;
    let mut j = 0;
    while i > 0 {
        if i & 1 != 0 {
            res ^= SOBOL_MATRIX[dim as usize][j as usize];
        }
        j += 1;
        i >>= 1;
    }
    let r = res as f32 * (1.0 / 0xffffffffu32 as f32);
    let tmp_rng = cmj_hash_simple(dim, rng);
    let shift = tmp_rng as f32 * (1.0 / 0xffffffffu32 as f32);
    r + shift - (r + shift).floor()
}
impl SobolSampler {
    pub fn new(_index: u32) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            dim: 0,
            index: 0,
            rotation: rng.gen(),
        }
    }
}
impl Sampler for SobolSampler {
    fn next1d(&mut self) -> Float {
        let r = sobol(self.dim, self.index, self.rotation);
        self.dim += 1;
        r as Float
    }
    // fn start_pixel(&mut self, px: &IVec2, res: &IVec2) {
    //     self.index = (px.x + px.y * res.x) as u32;
    //     self.dim = 0;
    //     let mut rng = rand::thread_rng();
    //     self.rotation = rng.gen();
    // }
    fn start_next_sample(&mut self) {
        self.index += 1;
        self.dim = 0;
    }
}
