use std::process::exit;
use std::sync::Arc;

use hexf::{hexf32, hexf64};
use lazy_static::lazy_static;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

use crate::data::bluenoise::{BLUE_NOISE_RESOLUTION, N_BLUE_NOISE_TEXTURES};
use crate::data::pmj02bn::N_PMJ02BN_SETS;
use crate::util::hash::xxhash32_4;
use crate::util::{is_power_of_four, log4u32, round_up_pow4};
use crate::{scene::Scene, *};
pub mod mcmc;
use crate::data::{bluenoise, pmj02bn};
use serde::{Deserialize, Serialize};

pub trait Sampler {
    fn next_1d(&self) -> Expr<f32>;
    fn next_2d(&self) -> Expr<Float2> {
        let u0 = self.next_1d();
        let u1 = self.next_1d();
        Float2::expr(u0, u1)
    }
    fn next_3d(&self) -> Expr<Float3> {
        let u0 = self.next_1d();
        let u12 = self.next_2d();
        Float3::expr(u0, u12.x, u12.y)
    }
    fn next_4d(&self) -> Expr<Float4> {
        let u01 = self.next_2d();
        let u23 = self.next_2d();
        Float4::expr(u01.x, u01.y, u23.x, u23.y)
    }
    fn is_metropolis(&self) -> bool {
        false
    }
    fn uniform(&self) -> Expr<f32> {
        self.next_1d()
    }
    fn uniform2(&self) -> Expr<Float2> {
        Float2::expr(self.uniform(), self.uniform())
    }
    fn uniform3(&self) -> Expr<Float3> {
        Float3::expr(self.uniform(), self.uniform(), self.uniform())
    }

    fn start(&self);
    fn clone_box(&self) -> Box<dyn Sampler>;
    // Don't write state to buffer when dropped
    fn forget(&self);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SampleStream {
    Pixel,
    Camera, // (filter + lens + wavelenths)
    Light,
    Bsdf,
    Roulette,
}
impl SampleStream {
    pub const fn dimension(&self) -> u32 {
        match self {
            SampleStream::Pixel => 2,
            SampleStream::Camera => 2 + 2 + 1,
            SampleStream::Light => 3,
            SampleStream::Bsdf => 3,
            SampleStream::Roulette => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
#[value_new(pub)]
pub struct Pcg32 {
    pub state: u64,
    pub inc: u64,
}
impl Pcg32 {
    pub const PCG32_DEFAULT_STATE: u64 = 0x853c49e6748fea9bu64;
    pub const PCG32_DEFAULT_STREAM: u64 = 0xda3e39cb94b95bdbu64;
    pub const PCG32_MULT: u64 = 0x5851f42d4c957f2du64;
}
impl Pcg32Var {
    #[tracked(crate = "luisa")]
    fn set_seq_offset(seq: Expr<u64>, seed: Expr<u64>) -> Var<Pcg32> {
        let pcg = Pcg32::new_expr(0, (seq << 1u64) | 1u64).var();
        pcg.gen_u32();
        *pcg.state += seed;
        pcg.gen_u32();
        pcg
    }
    pub fn new_seq_offset(seq: Expr<u64>, offset: Expr<u64>) -> Var<Pcg32> {
        Self::set_seq_offset(seq, offset)
    }
    pub fn new_seq(seq: Expr<u64>) -> Var<Pcg32> {
        Self::set_seq_offset(seq, util::mix_bits(seq))
    }
    pub fn gen_u32(&self) -> Expr<u32> {
        lazy_static! {
            static ref PCG_GEN_U32: Callable<fn(Var<Pcg32>) -> Expr<u32>> =
                Callable::<fn(Var<Pcg32>) -> Expr<u32>>::new_static(track!(|pcg: Var<Pcg32>| {
                    let old_state = **pcg.state;
                    *pcg.state = old_state * Pcg32::PCG32_MULT + pcg.inc;
                    let xor_shifted: Expr<u32> = (((old_state >> 18) ^ old_state) >> 27).cast_u32();
                    let rot = (old_state >> 59).cast_u32();
                    (xor_shifted >> rot) | (xor_shifted << ((!rot + 1) & 31))
                }));
        }
        PCG_GEN_U32.call(self.self_)
    }
    #[tracked(crate = "luisa")]
    pub fn advance(&self, idelta: Expr<i64>) {
        let cur_mult = Pcg32::PCG32_MULT.var();
        let cur_plus = self.inc.var();
        let acc_mult = 1u64.var();
        let acc_plus = 0u64.var();
        let delta = idelta.cast_u64().var();
        while delta > 0 {
            if (delta & 1) != 0 {
                *acc_mult *= cur_mult;
                *acc_plus = acc_plus + cur_mult + cur_plus;
            }
            *cur_plus = (cur_mult + 1) * cur_plus;
            *cur_mult *= cur_mult;
            *delta >>= 1u64;
        }
        *self.state = acc_mult * self.state + acc_plus;
    }
}
#[tracked(crate = "luisa")]
pub fn init_pcg32_buffer(device: Device, count: usize) -> Buffer<Pcg32> {
    let buffer = device.create_buffer(count);
    let mut rng = thread_rng();
    let seeds = device.create_buffer_from_fn(count, |_| rng.gen::<u64>());
    Kernel::<fn()>::new(&device, &|| {
        let i = dispatch_id().x;
        let seed = seeds.read(i);
        let pcg = Pcg32Var::new_seq_offset(i.cast_u64(), seed);
        buffer.write(i, pcg.load());
    })
    .dispatch([count.try_into().unwrap(), 1, 1]);
    buffer
}
#[tracked(crate = "luisa")]
pub fn init_pcg32_buffer_with_seed(device: Device, count: usize, seed: u64) -> Buffer<Pcg32> {
    let buffer = device.create_buffer(count);
    let mut rng = StdRng::seed_from_u64(seed);
    let seeds = device.create_buffer_from_fn(count, |_| rng.gen::<u64>());
    Kernel::<fn()>::new(&device, &|| {
        let i = dispatch_id().x;
        let seed = seeds.read(i);
        let pcg = Pcg32Var::new_seq_offset(i.cast_u64(), seed);
        buffer.write(i, pcg.load());
    })
    .dispatch([count.try_into().unwrap(), 1, 1]);
    buffer
}
pub struct IndependentSampler {
    pub state: Var<Pcg32>,
    index: Expr<u32>,
    states: Option<Arc<Buffer<Pcg32>>>,
    forget: Var<bool>,
    dim: Var<u32>,
}
impl Drop for IndependentSampler {
    #[tracked(crate = "luisa")]
    fn drop(&mut self) {
        if let Some(states) = self.states.take() {
            self.state.advance(-self.dim.as_i64());
            if !self.forget {
                states.write(self.index, self.state);
            }
        }
    }
}
impl IndependentSampler {
    pub fn from_pcg32(state: Var<Pcg32>) -> Self {
        Self {
            state,
            index: 0u32.expr(),
            states: None,
            forget: false.var(),
            dim: 0u32.var(),
        }
    }
    /// How come you use more than that?
    pub const MAX_DIM_PER_SPP: u32 = 16384;
}
impl Sampler for IndependentSampler {
    #[tracked(crate = "luisa")]
    fn next_1d(&self) -> Expr<f32> {
        *self.dim += 1;
        let n = self.state.gen_u32();
        n.cast_f32() * ((1.0 / u32::MAX as f64) as f32)
    }
    fn start(&self) {
        if let Some(_) = &self.states {
            self.state.advance((Self::MAX_DIM_PER_SPP as i64).expr());
        }
    }
    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(Self {
            state: (*self.state).var(),
            index: self.index,
            states: self.states.clone(),
            forget: (*self.forget).var(),
            dim: (*self.dim).var(),
        })
    }
    #[tracked(crate = "luisa")]
    fn forget(&self) {
        *self.forget = true.expr();
    }
}
#[derive(Copy, Clone, Aggregate)]
#[luisa(crate = "luisa")]
pub struct PrimarySample {
    pub values: VLArrayVar<f32>,
}
impl PrimarySample {
    #[tracked(crate = "luisa")]
    pub fn clamped(&self) -> Self {
        let values = VLArrayVar::zero(self.values.len());
        for_range(0u64.expr()..values.len_expr(), |i| {
            let x = self.values.read(i);
            values.write(i, x - x.floor());
        });
        Self { values }
    }
    #[tracked(crate = "luisa")]
    pub fn new(len: usize, sampler: &dyn Sampler) -> Self {
        let values = VLArrayVar::zero(len);
        for_range(0u64.expr()..values.len_expr(), |i| {
            values.write(i, sampler.next_1d());
        });
        Self { values }
    }
}
pub struct IndependentReplaySampler<'a> {
    pub base: &'a IndependentSampler,
    pub sample: PrimarySample,
    pub cur_dim: Var<u32>,
}
impl<'a> IndependentReplaySampler<'a> {
    pub fn new(base: &'a IndependentSampler, sample: PrimarySample) -> Self {
        Self {
            base,
            sample,
            cur_dim: 0u32.var(),
        }
    }
}

impl<'a> Sampler for IndependentReplaySampler<'a> {
    #[tracked(crate = "luisa")]
    fn next_1d(&self) -> Expr<f32> {
        if self
            .cur_dim
            .load()
            .lt(self.sample.values.len_expr().cast_u32())
        {
            let ret = self.sample.values.read(self.cur_dim.load());
            *self.cur_dim += 1;
            ret
        } else {
            *self.cur_dim += 1;
            self.base.next_1d()
        }
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
    fn forget(&self) {}
    fn clone_box(&self) -> Box<dyn Sampler> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum SamplerConfig {
    #[serde(rename = "independent")]
    Independent { seed: u64 },
    #[serde(rename = "pmj02bn")]
    Pmj02Bn { seed: u64 },
}
impl Default for SamplerConfig {
    fn default() -> Self {
        Self::Independent { seed: 0 }
    }
}
pub trait SamplerCreator {
    fn create(&self, pixel: Expr<Uint2>) -> Box<dyn Sampler>;
}
pub struct IndependentSamplerCreator {
    states: Arc<Buffer<Pcg32>>,
    resolution: Uint2,
}
impl IndependentSamplerCreator {
    pub fn new(device: Device, resolution: Uint2, seed: u64) -> Self {
        Self {
            states: Arc::new(init_pcg32_buffer_with_seed(
                device,
                resolution.x as usize * resolution.y as usize,
                seed,
            )),
            resolution,
        }
    }
}
impl SamplerCreator for IndependentSamplerCreator {
    #[tracked(crate = "luisa")]
    fn create(&self, pixel: Expr<Uint2>) -> Box<dyn Sampler> {
        let i = pixel.x + pixel.y * self.resolution.x;
        let state = self.states.read(i).var();
        Box::new(IndependentSampler {
            state,
            index: i,
            forget: false.var(),
            states: Some(self.states.clone()),
            dim: 0u32.var(),
        })
    }
}
#[allow(dead_code)]
pub struct Pmj02BnSamplerCreator {
    device: Device,
    pmj02bn_samples: Arc<Buffer<u32>>,
    pixel_samples: Arc<Buffer<Float2>>,
    bluenoise_textures: Arc<BindlessArray>,
    states: Arc<Buffer<Pmj02BnState>>,
    resolution: Uint2,
    spp: u32,
    seed: u32,
    pixel_tile_size: u32,
}
pub fn pmj02bn_sample_host(mut set_index: u32, mut sample_index: u32) -> Float2 {
    set_index %= pmj02bn::N_PMJ02BN_SETS as u32;
    assert!(sample_index < pmj02bn::N_PMJ02BN_SAMPLES as u32);
    sample_index %= pmj02bn::N_PMJ02BN_SAMPLES as u32;
    Float2::new(
        (pmj02bn::PMJ02BN_SAMPLES[set_index as usize][sample_index as usize][0] as f64
            * (1.0 / u32::MAX as f64)) as f32,
        (pmj02bn::PMJ02BN_SAMPLES[set_index as usize][sample_index as usize][1] as f64
            * (1.0 / u32::MAX as f64)) as f32,
    )
}
#[tracked(crate = "luisa")]
pub fn pmj02bn_sample(
    pmj02bn_samples: &Buffer<u32>,
    mut set_index: Expr<u32>,
    mut sample_index: Expr<u32>,
) -> Expr<Float2> {
    set_index = set_index % pmj02bn::N_PMJ02BN_SETS as u32;
    if debug_mode() {
        lc_assert!(sample_index.lt(pmj02bn::N_PMJ02BN_SAMPLES as u32));
    }
    sample_index = sample_index % pmj02bn::N_PMJ02BN_SAMPLES as u32;
    let i = pmj02bn::N_PMJ02BN_SAMPLES as u32 * set_index + sample_index;
    Float2::expr(
        pmj02bn_samples.read(i * 2).cast_f32() * hexf32!("0x1p-32"),
        pmj02bn_samples.read(i * 2 + 1).cast_f32() * hexf32!("0x1p-32"),
    )
}
impl Pmj02BnSamplerCreator {
    pub fn new(device: Device, resolution: Uint2, seed: u32, spp: u32) -> Self {
        if !is_power_of_four(spp) {
            log::warn!("Pmj02BnSampler results are best with power-of-4 samples per pixel (1, 4, 16, 64, ...)");
        }
        if spp > pmj02bn::N_PMJ02BN_SAMPLES as u32 {
            log::error!(
                "Pmj02BnSampler supports up to {} spp",
                pmj02bn::N_PMJ02BN_SAMPLES
            );
            exit(-1);
        }
        let mut w = spp - 1;
        w |= w >> 1;
        w |= w >> 2;
        w |= w >> 4;
        w |= w >> 8;
        w |= w >> 16;
        let pixel_tile_size =
            1u32 << (log4u32(pmj02bn::N_PMJ02BN_SAMPLES as u32) - log4u32(round_up_pow4(spp)));
        let n_pixel_samples = pixel_tile_size.pow(2) * spp;
        let mut pixel_samples = vec![Float2::new(0.0, 0.0); n_pixel_samples as usize];
        let mut n_stored = vec![0u32; pixel_tile_size.pow(2) as usize];
        for i in 0..pmj02bn::N_PMJ02BN_SAMPLES {
            let mut p: glam::Vec2 = pmj02bn_sample_host(0, i as u32).into();
            p *= pixel_tile_size as f32;
            let pixel_offset = p.x.floor() as u32 + p.y.floor() as u32 * pixel_tile_size;
            if n_stored[pixel_offset as usize] == spp {
                assert!(!is_power_of_four(spp));
                continue;
            }
            let sample_offset = pixel_offset * spp + n_stored[pixel_offset as usize];
            assert!(pixel_samples[sample_offset as usize].x == 0.0);
            assert!(pixel_samples[sample_offset as usize].y == 0.0);
            pixel_samples[sample_offset as usize] = (p - p.floor()).into();
            n_stored[pixel_offset as usize] += 1;
        }
        for i in 0..n_stored.len() {
            assert_eq!(spp, n_stored[i as usize], "i: {}", i);
        }
        let pixel_samples = Arc::new(device.create_buffer_from_slice(&pixel_samples));
        let pmj02bn_samples = Arc::new(device.create_buffer_from_slice(unsafe {
            let ptr = pmj02bn::PMJ02BN_SAMPLES.as_ptr() as *const u32;
            let len = std::mem::size_of_val(&pmj02bn::PMJ02BN_SAMPLES) / std::mem::size_of::<u32>();
            assert_eq!(
                len,
                pmj02bn::N_PMJ02BN_SETS * pmj02bn::N_PMJ02BN_SAMPLES * 2
            );
            std::slice::from_raw_parts(ptr, len)
        }));
        let bluenoise_textures = device.create_bindless_array(N_BLUE_NOISE_TEXTURES);
        for i in 0..N_BLUE_NOISE_TEXTURES {
            let tex = device.create_tex2d::<f32>(
                PixelStorage::Short1,
                BLUE_NOISE_RESOLUTION as u32,
                BLUE_NOISE_RESOLUTION as u32,
                1,
            );
            tex.view(0).copy_from(unsafe {
                let ptr = bluenoise::BLUE_NOISE_TEXTURES[i].as_ptr() as *const u16;
                let len = std::mem::size_of_val(&bluenoise::BLUE_NOISE_TEXTURES[i])
                    / std::mem::size_of::<u16>();
                assert_eq!(len, BLUE_NOISE_RESOLUTION * BLUE_NOISE_RESOLUTION);
                std::slice::from_raw_parts(ptr, len)
            });
            bluenoise_textures.set_tex2d(
                i as usize,
                &tex,
                TextureSampler {
                    filter: SamplerFilter::Point,
                    address: SamplerAddress::Repeat,
                },
            );
        }
        let states = Arc::new(device.create_buffer_from_fn(
            resolution.x as usize * resolution.y as usize,
            |i| {
                let x = (i % resolution.x as usize) as u32;
                let y = (i / resolution.x as usize) as u32;
                Pmj02BnState {
                    seed,
                    dim: 0,
                    pixel: Uint2::new(x, y),
                    sample_index: u32::MAX,
                    spp,
                    w,
                }
            },
        ));
        Self {
            device,
            pmj02bn_samples,
            pixel_samples,
            resolution,
            spp,
            seed,
            pixel_tile_size,
            states,
            bluenoise_textures: Arc::new(bluenoise_textures),
        }
    }
}

// see https://github.com/LuisaGroup/LuisaRender/blob/next/src/samplers/pmj02bn.cpp#L56
lazy_static! {
    static ref PERMUTE_ELEMENT: Callable<fn(Expr<u32>, Expr<u32>, Expr<u32>, Expr<u32>) -> Expr<u32>> =
        Callable::<fn(Expr<u32>, Expr<u32>, Expr<u32>, Expr<u32>) -> Expr<u32>>::new_static(
            |i: Expr<u32>, l: Expr<u32>, w: Expr<u32>, p: Expr<u32>| {
                track!({
                    let i = i.var();
                    loop {
                        *i ^= p;
                        *i *= 0xe170893du32;
                        *i ^= p >> 16u32;
                        *i ^= (i & w) >> 4u32;
                        *i ^= p >> 8u32;
                        *i *= 0x0929eb3fu32;
                        *i ^= p >> 23u32;
                        *i ^= (i & w) >> 1u32;
                        *i *= 1 | p >> 27u32;
                        *i *= 0x6935fa69u32;
                        *i ^= (i & w) >> 11u32;
                        *i *= 0x74dcb303u32;
                        *i ^= (i & w) >> 2u32;
                        *i *= 0x9e501cc3u32;
                        *i ^= (i & w) >> 2u32;
                        *i *= 0xc860a3dfu32;
                        *i &= w;
                        *i ^= i >> 5u32;
                        if i < l {
                            break;
                        }
                    }
                    (i + p) % l
                })
            }
        );
}
fn permute_element(i: Expr<u32>, l: Expr<u32>, w: Expr<u32>, p: Expr<u32>) -> Expr<u32> {
    PERMUTE_ELEMENT.call(i, l, w, p)
}
#[derive(Clone, Copy, Value, Debug)]
#[luisa(crate = "luisa")]
#[repr(C)]
struct Pmj02BnState {
    seed: u32,
    dim: u32,
    pixel: Uint2,
    sample_index: u32,
    spp: u32,
    w: u32,
}
pub struct Pmj02BnSampler {
    pmj02bn_samples: Arc<Buffer<u32>>,
    pixel_samples: Arc<Buffer<Float2>>,
    bluenoise_textures: Arc<BindlessArray>,
    i: Expr<u32>,
    state: Var<Pmj02BnState>,
    states: Arc<Buffer<Pmj02BnState>>,
    forget: Var<bool>,
    next_1d: Arc<DynCallable<fn(Var<Pmj02BnState>) -> Expr<f32>>>,
    next_2d: Arc<DynCallable<fn(Var<Pmj02BnState>) -> Expr<Float2>>>,
}
impl Pmj02BnSampler {
    #[tracked(crate = "luisa")]
    fn new(
        device: &Device,
        pmj02bn_samples: Arc<Buffer<u32>>,
        pixel_samples: Arc<Buffer<Float2>>,
        bluenoise_textures: Arc<BindlessArray>,
        states: Arc<Buffer<Pmj02BnState>>,
        i: Expr<u32>,
    ) -> Self {
        let bluenoise =
            |bluenoise_textures: &Arc<BindlessArray>, tex_index: Expr<u32>, p: Expr<Uint2>| {
                let uv = p.yx() % BLUE_NOISE_RESOLUTION as u32;
                bluenoise_textures
                    .var()
                    .tex2d(tex_index % N_BLUE_NOISE_TEXTURES as u32)
                    .read(uv)
                    .x
            };
        let next_1d = {
            let bluenoise_textures = bluenoise_textures.clone();
            DynCallable::<fn(Var<Pmj02BnState>) -> Expr<f32>>::new(
                &device,
                track!(Box::new(move |state: Var<Pmj02BnState>| {
                    let hash = xxhash32_4(Uint4::expr(
                        state.pixel.x,
                        state.pixel.y,
                        state.dim,
                        state.seed,
                    ));
                    let index = permute_element(**state.sample_index, **state.spp, **state.w, hash);
                    let delta = bluenoise(&bluenoise_textures, **state.dim, **state.pixel);
                    // if (state.pixel == Uint2::expr(255, 224)).all() {
                    //     device_log!(
                    //         "px: {} index: {}, dim:{} ",
                    //         state.pixel,
                    //         state.sample_index,
                    //         state.dim
                    //     );
                    // }
                    *state.dim += 1;
                    ((index.cast_f32() + delta) / state.spp.cast_f32()).min_(ONE_MINUS_EPSILON)
                })),
            )
        };
        // let permute_element = permute_element_.clone();
        let next_2d = {
            let pmj02bn_samples = pmj02bn_samples.clone();
            let bluenoise_textures = bluenoise_textures.clone();
            DynCallable::<fn(Var<Pmj02BnState>) -> Expr<Float2>>::new(
                &device,
                track!(Box::new(move |state: Var<Pmj02BnState>| {
                    let index = state.sample_index.var();
                    let dim = **state.dim;
                    let pmj_instance = dim / 2;
                    if pmj_instance >= N_PMJ02BN_SETS as u32 {
                        let hash = xxhash32_4(Uint4::expr(
                            state.pixel.x,
                            state.pixel.y,
                            state.dim,
                            state.seed,
                        ));
                        *index =
                            permute_element(**state.sample_index, **state.spp, **state.w, hash);
                    };
                    let u = pmj02bn_sample(&pmj02bn_samples, pmj_instance, **index);

                    let dx = bluenoise(&bluenoise_textures, dim, **state.pixel);
                    let dy = bluenoise(&bluenoise_textures, dim + 1, **state.pixel);
                    let delta = Float2::expr(dx, dy);
                    // let delta = Float2::expr(0.2,0.3);
                    // if (state.pixel == Uint2::expr(255, 224)).all() {
                    //     let uf = (u + delta).fract();
                    //     device_log!(
                    //         "px: {} index: {}, dim:{} u: {}, delta: {}, fract: {}",
                    //         state.pixel,
                    //         state.sample_index,
                    //         state.dim,
                    //         u,
                    //         delta,
                    //         uf
                    //     );
                    // }
                    let u = u + delta;
                    *state.dim += 2;
                    let u = u.fract();
                    let ux = u.x.min_(ONE_MINUS_EPSILON);
                    let uy = u.y.min_(ONE_MINUS_EPSILON);
                    Float2::expr(ux, uy)
                })),
            )
        };
        Self {
            pmj02bn_samples,
            pixel_samples,
            bluenoise_textures,
            state: states.read(i).var(),
            forget: false.var(),
            states,
            i,
            next_1d: Arc::new(next_1d),
            next_2d: Arc::new(next_2d),
        }
    }
}
impl Drop for Pmj02BnSampler {
    #[tracked(crate = "luisa")]
    fn drop(&mut self) {
        if !self.forget {
            let i = self.i;
            self.states.write(i, self.state);
        }
    }
}
impl Sampler for Pmj02BnSampler {
    fn next_1d(&self) -> Expr<f32> {
        (self.next_1d).call(self.state)
    }
    fn next_2d(&self) -> Expr<Float2> {
        (self.next_2d).call(self.state)
        // Float2::expr(
        //     self.next_1d(),
        // self.next_1d())
    }
    #[tracked(crate = "luisa")]
    fn start(&self) {
        // thank you, pmj
        // starting from <4 produces weird samples
        *self.state.dim = 4u32.expr();
        if self.state.sample_index.eq(u32::MAX) {
            *self.state.sample_index = 0;
        } else {
            *self.state.sample_index += 1;
        };
        if debug_mode() {
            lc_assert!(self.state.sample_index.lt(self.state.spp));
        };
    }
    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(Self {
            pmj02bn_samples: self.pmj02bn_samples.clone(),
            pixel_samples: self.pixel_samples.clone(),
            bluenoise_textures: self.bluenoise_textures.clone(),
            i: self.i,
            state: self.state.var(),
            states: self.states.clone(),
            forget: self.forget.var(),
            next_1d: self.next_1d.clone(),
            next_2d: self.next_2d.clone(),
        })
    }
    #[tracked(crate = "luisa")]
    fn forget(&self) {
        *self.forget = true.expr();
    }
}
impl SamplerCreator for Pmj02BnSamplerCreator {
    #[tracked(crate = "luisa")]
    fn create(&self, pixel: Expr<Uint2>) -> Box<dyn Sampler> {
        let i = pixel.x + pixel.y * self.resolution.x;
        Box::new(Pmj02BnSampler::new(
            &self.device,
            self.pmj02bn_samples.clone(),
            self.pixel_samples.clone(),
            self.bluenoise_textures.clone(),
            self.states.clone(),
            i,
        ))
    }
}
impl SamplerConfig {
    pub fn creator(&self, device: Device, scene: &Scene, spp: u32) -> Box<dyn SamplerCreator> {
        match self {
            Self::Independent { seed } => Box::new(IndependentSamplerCreator::new(
                device,
                scene.camera.resolution(),
                *seed,
            )),
            Self::Pmj02Bn { seed } => Box::new(Pmj02BnSamplerCreator::new(
                device,
                scene.camera.resolution(),
                *seed as u32,
                spp,
            )),
        }
    }
}
