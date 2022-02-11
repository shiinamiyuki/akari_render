use std::convert::TryInto;

use crate::*;

use akari_const::SOBOL_MATRIX;
use util::erf_inv;
pub trait Sampler: Sync + Send {
    // fn start_pixel(&mut self, px: &IVec2, res: &IVec2);
    fn start_next_sample(&mut self);
    fn next1d(&mut self) -> f32;
    fn next2d(&mut self) -> Vec2 {
        vec2(self.next1d(), self.next1d())
    }
    fn next3d(&mut self) -> Vec3 {
        vec3(self.next1d(), self.next1d(), self.next1d())
    }
}

pub struct Lcg {
    state: u32,
}
impl Lcg {
    const INC: u32 = 12345;
    const MULTIPILER: u32 = 1103515245;
    pub fn new(seed: u32) -> Self {
        let mut r = Self {
            state: seed + Self::INC,
        };
        let _ = r.lcg32();
        r
    }
    pub fn lcg32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPILER)
            .wrapping_add(Self::INC);
        self.state
    }
}
pub struct Pcg {
    state: u64,
}
impl Pcg {
    const MULTIPILER: u64 = 6364136223846793005;
    const INC: u64 = 1442695040888963407;

    pub fn pcg32(&mut self) -> u32 {
        let mut x = self.state;
        let count = x >> 59;
        self.state = x.wrapping_mul(Self::MULTIPILER).wrapping_add(Self::INC); //x * Self::MULTIPILER + Self::INC;
        x ^= x >> 18;
        ((x >> 27) as u32).rotate_right(count as u32)
    }
    pub fn pcg64(&mut self) -> u64 {
        self.pcg32() as u64 | ((self.pcg32() as u64) << 32)
    }
    pub fn new(seed: u64) -> Self {
        let mut r = Self {
            state: seed.wrapping_add(Self::INC),
        };
        let _ = r.pcg32();
        r
    }
}
pub struct PCGSampler {
    pub rng: Pcg,
}
impl PCGSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Pcg::new(seed),
        }
    }
}
impl Sampler for PCGSampler {
    fn next1d(&mut self) -> f32 {
        (self.rng.pcg32() as f64 / (std::u32::MAX as f64)) as f32
    }
    // fn start_pixel(&mut self, px: &IVec2, res: &IVec2) {}
    fn start_next_sample(&mut self) {}
}
#[derive(Clone, Copy)]
pub struct SobolSampler {
    pub(crate) dim: u32,
    pub(crate) rotation: u32,
    pub(crate) index: u32,
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
    let shift = tmp_rng as f32 * (1.0 / 0xffffffffu32 as f64) as f32;
    r + shift - (r + shift).floor()
}
impl SobolSampler {
    pub fn new(seed: u64) -> Self {
        let mut pcg = Pcg::new(seed);
        Self {
            dim: 0,
            index: 0,
            rotation: pcg.pcg32(),
        }
    }
}
impl Sampler for SobolSampler {
    fn next1d(&mut self) -> f32 {
        let r = sobol(self.dim, self.index, self.rotation);
        self.dim += 1;
        r as f32
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

pub struct ReplaySampler<S> {
    x: Vec<f32>,
    base: S,
    index: usize,
    replaying: bool,
}
impl<S: Sampler> ReplaySampler<S> {
    pub fn new(s: S) -> Self {
        Self {
            x: vec![],
            base: s,
            index: 0,
            replaying: false,
        }
    }
    pub fn ensure_ready(&mut self, i: usize) {
        while i >= self.x.len() {
            self.x.push(self.base.next1d());
        }
    }
    pub fn replay(&mut self) {
        self.replaying = true;
    }
    pub fn reset(&mut self) {
        self.replaying = false;
    }
    pub fn x(&mut self, index: usize) -> f32 {
        self.ensure_ready(index);
        self.x[index]
    }
}
impl<S: Sampler> Sampler for ReplaySampler<S> {
    fn start_next_sample(&mut self) {
        self.index = 0;
        if !self.replaying {
            self.base.start_next_sample();
            self.x.clear();
        }
    }

    fn next1d(&mut self) -> f32 {
        self.ensure_ready(self.index);
        let x = self.x[self.index];
        self.index += 1;
        x
    }
}

pub struct PrimarySample {
    pub value: f32,
    pub backup: f32,
    pub last_modified: usize,
    pub modified_backup: usize,
}
impl PrimarySample {
    pub fn new(x: f32) -> Self {
        Self {
            value: x,
            backup: x,
            last_modified: 0,
            modified_backup: 0,
        }
    }
    pub fn backup(&mut self) {
        self.backup = self.value;
        self.modified_backup = self.last_modified;
    }
    pub fn restore(&mut self) {
        self.value = self.backup;
        self.last_modified = self.modified_backup;
    }
}

#[repr(align(64))]
pub struct MltSampler {
    pub samples: Vec<PrimarySample>,
    pub rng: PCGSampler,
    pub large_step: bool,
    pub cur_iteration: isize,
    pub dimension: usize,
    pub last_large_iteration: usize,
}
const SIGMA: f32 = 0.01;
impl MltSampler {
    pub fn from_replay<S>(replay: &ReplaySampler<S>, seed: u64) -> Self {
        let mut s = MltSampler::new(seed);
        for x in &replay.x {
            s.samples.push(PrimarySample::new(*x))
        }
        s
    }
    pub fn new(seed: u64) -> Self {
        Self {
            samples: vec![],
            rng: PCGSampler {
                rng: Pcg::new(seed),
            },
            large_step: false,
            last_large_iteration: 0,
            cur_iteration: 0,
            dimension: 0,
        }
    }

    pub fn update(&mut self, i: usize) {
        let x = &mut self.samples[i];
        if x.last_modified < self.last_large_iteration {
            x.value = self.rng.next1d();
            x.last_modified = self.last_large_iteration;
        }
        x.backup();
        if self.large_step {
            x.value = self.rng.next1d();
        } else {
            let n_small: usize = self.cur_iteration as usize - x.last_modified;
            let normal_sample = (2.0 as f32).sqrt() * erf_inv(2.0 * self.rng.next1d() - 1.0);
            let err_sigma = SIGMA * (n_small as f32).sqrt();
            x.value += normal_sample * err_sigma;
            x.value -= x.value.floor();
        }
        x.last_modified = self.cur_iteration.try_into().unwrap();
    }
    pub fn accept(&mut self) {
        if self.large_step {
            self.last_large_iteration = self.cur_iteration as usize;
        }
    }
    pub fn reject(&mut self) {
        for x in &mut self.samples {
            if x.last_modified == self.cur_iteration as usize {
                x.restore();
            }
        }
        self.cur_iteration -= 1;
    }
    pub fn start_new_iteration(&mut self, is_large_step: bool) {
        self.cur_iteration += 1;
        self.large_step = is_large_step;
        self.dimension = 0;
    }
    pub fn reseed(&mut self, seed: u64) {
        self.rng = PCGSampler::new(seed);
    }
}
impl Sampler for MltSampler {
    fn start_next_sample(&mut self) {
        panic!("call start_new_iteration instead for mlt sampler");
    }

    fn next1d(&mut self) -> f32 {
        let idx = self.dimension;
        self.dimension += 1;
        while idx >= self.samples.len() {
            self.samples.push(PrimarySample::new(self.rng.next1d()));
        }
        self.update(idx);
        self.samples[idx].value
    }
}
