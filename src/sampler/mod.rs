use std::process::exit;
use std::sync::Arc;

use hexf::{hexf32, hexf64};
use lazy_static::lazy_static;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

use crate::data::bluenoise::{BLUE_NOISE_RESOLUTION, N_BLUE_NOISE_TEXTURES};
use crate::data::pmj02bn::N_PMJ02BN_SETS;
use crate::util::hash::xxhash32_4;
use crate::util::{is_power_of_four, log4u32};
use crate::{scene::Scene, *};
pub mod mcmc;
use crate::data::{bluenoise, pmj02bn};
use serde::{Deserialize, Serialize};

pub trait Sampler {
    fn next_1d(&self) -> Float;
    fn next_2d(&self) -> Expr<Float2> {
        let u0 = self.next_1d();
        let u1 = self.next_1d();
        make_float2(u0, u1)
    }
    fn next_3d(&self) -> Expr<Float3> {
        let u0 = self.next_1d();
        let u12 = self.next_2d();
        make_float3(u0, u12.x(), u12.y())
    }
    fn next_4d(&self) -> Expr<Float4> {
        let u01 = self.next_2d();
        let u23 = self.next_2d();
        make_float4(u01.x(), u01.y(), u23.x(), u23.y())
    }
    fn is_metropolis(&self) -> bool {
        false
    }
    fn uniform(&self) -> Expr<f32> {
        self.next_1d()
    }
    fn uniform2(&self) -> Expr<Float2> {
        make_float2(self.next_1d(), self.next_1d())
    }
    fn uniform3(&self) -> Expr<Float3> {
        make_float3(self.next_1d(), self.next_1d(), self.next_1d())
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
#[repr(C)]
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
    fn set_seq_offset(seq: Expr<u64>, seed: Expr<u64>) -> Self {
        let pcg = var!(Pcg32, Pcg32Expr::new(0, (seq << 1) | 1));
        pcg.gen_u32();
        pcg.state().store(pcg.state().load() + seed);
        pcg.gen_u32();
        pcg
    }
    pub fn new_seq_offset(seq: Expr<u64>, offset: Expr<u64>) -> Self {
        Self::set_seq_offset(seq, offset)
    }
    pub fn new_seq(seq: Expr<u64>) -> Self {
        Self::set_seq_offset(seq, util::mix_bits(seq))
    }
    pub fn gen_u32(&self) -> Expr<u32> {
        lazy_static! {
            static ref PCG_GEN_U32: Callable<(Var<Pcg32>,), Expr<u32>> =
                create_static_callable::<(Var<Pcg32>,), Expr<u32>>(|pcg: Var<Pcg32>| {
                    let old_state = pcg.state().load();
                    pcg.state()
                        .store(old_state * Pcg32::PCG32_MULT + pcg.inc().load());
                    let xor_shifted: Expr<u32> = (((old_state >> 18) ^ old_state) >> 27).uint();
                    let rot = (old_state >> 59).uint();
                    (xor_shifted >> rot) | (xor_shifted << ((!rot + 1) & 31))
                });
        }
        PCG_GEN_U32.call(*self)
    }
}
pub fn init_pcg32_buffer(device: Device, count: usize) -> Buffer<Pcg32> {
    let buffer = device.create_buffer(count);
    let mut rng = thread_rng();
    let seeds = device.create_buffer_from_fn(count, |_| rng.gen::<u64>());
    device
        .create_kernel::<()>(&|| {
            let i = dispatch_id().x();
            let seed = seeds.var().read(i);
            let pcg = Pcg32Var::new_seq_offset(i.ulong(), seed);
            buffer.var().write(i, pcg.load());
        })
        .dispatch([count.try_into().unwrap(), 1, 1]);
    buffer
}
pub fn init_pcg32_buffer_with_seed(device: Device, count: usize, seed: u64) -> Buffer<Pcg32> {
    let buffer = device.create_buffer(count);
    let mut rng = StdRng::seed_from_u64(seed);
    let seeds = device.create_buffer_from_fn(count, |_| rng.gen::<u64>());
    device
        .create_kernel::<()>(&|| {
            let i = dispatch_id().x();
            let seed = seeds.var().read(i);
            let pcg = Pcg32Var::new_seq_offset(i.ulong(), seed);
            buffer.var().write(i, pcg.load());
        })
        .dispatch([count.try_into().unwrap(), 1, 1]);
    buffer
}
pub struct IndependentSampler {
    pub state: Pcg32Var,
    index: Expr<u32>,
    states: Option<Arc<Buffer<Pcg32>>>,
    forget: Var<bool>,
}
impl Drop for IndependentSampler {
    fn drop(&mut self) {
        if let Some(states) = self.states.take() {
            if_!(!*self.forget, {
                states.var().write(self.index, *self.state);
            });
        }
    }
}
impl IndependentSampler {
    pub fn from_pcg32(state: Pcg32Var) -> Self {
        Self {
            state,
            index: const_(0u32),
            states: None,
            forget: var!(bool, false),
        }
    }
}
impl Sampler for IndependentSampler {
    fn next_1d(&self) -> Float {
        let n = self.state.gen_u32();
        n.float() * ((1.0 / u32::MAX as f64) as f32)
    }
    fn start(&self) {}
    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(Self {
            state: var!(Pcg32, *self.state),
            index: self.index,
            states: self.states.clone(),
            forget: var!(bool, *self.forget),
        })
    }
    fn forget(&self) {
        *self.forget.get_mut() = true.into();
    }
}
#[derive(Copy, Clone, Aggregate)]
pub struct PrimarySample {
    pub values: VLArrayVar<f32>,
}
impl PrimarySample {
    pub fn clamped(&self) -> Self {
        let values = VLArrayVar::zero(self.values.static_len());
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
            let x = self.values.read(i);
            values.write(i, x - x.floor());
        });
        Self { values }
    }
    pub fn new(len: usize, sampler: &dyn Sampler) -> Self {
        let values = VLArrayVar::zero(len);
        for_range(const_(0)..values.len().int(), |i| {
            let i = i.uint();
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
            cur_dim: var!(u32, 0),
        }
    }
}

impl<'a> Sampler for IndependentReplaySampler<'a> {
    fn next_1d(&self) -> Float {
        if_!(self.cur_dim.load().cmplt(self.sample.values.len()), {
            let ret = self.sample.values.read(self.cur_dim.load());
            self.cur_dim.store(self.cur_dim.load() + 1);
            ret
        }, else {
            self.cur_dim.store(self.cur_dim.load() + 1);
            self.base.next_1d()
        })
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
    fn create(&self, pixel: Expr<Uint2>) -> Box<dyn Sampler> {
        let i = pixel.x() + pixel.y() * self.resolution.x;
        let state = var!(Pcg32, self.states.read(i));
        Box::new(IndependentSampler {
            state,
            index: i,
            forget: var!(bool, false),
            states: Some(self.states.clone()),
        })
    }
}

pub struct Pmj02BnSamplerCreator {
    device: Device,
    pmj02bn_samples: Arc<Buffer<f32>>,
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
            * hexf64!("0x1p-32")) as f32,
        (pmj02bn::PMJ02BN_SAMPLES[set_index as usize][sample_index as usize][1] as f64
            * hexf64!("0x1p-32")) as f32,
    )
}
pub fn pmj02bn_sample(
    pmj02bn_samples: &Buffer<f32>,
    mut set_index: Expr<u32>,
    mut sample_index: Expr<u32>,
) -> Expr<Float2> {
    set_index %= pmj02bn::N_PMJ02BN_SETS as u32;
    lc_assert!(sample_index.cmplt(pmj02bn::N_PMJ02BN_SAMPLES as u32));
    sample_index %= pmj02bn::N_PMJ02BN_SAMPLES as u32;
    let i = pmj02bn::N_PMJ02BN_SAMPLES as u32 * set_index + sample_index;
    make_float2(
        pmj02bn_samples.var().read(i * 2) * hexf32!("0x1p-32"),
        pmj02bn_samples.var().read(i * 2 + 1) * hexf32!("0x1p-32"),
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
        let pixel_tile_size = 1u32 << (log4u32(pmj02bn::N_PMJ02BN_SAMPLES as u32) - log4u32(spp));
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
            let ptr = pmj02bn::PMJ02BN_SAMPLES.as_ptr() as *const f32;
            let len = std::mem::size_of_val(&pmj02bn::PMJ02BN_SAMPLES) / std::mem::size_of::<f32>();
            assert_eq!(
                len,
                pmj02bn::N_PMJ02BN_SETS * pmj02bn::N_PMJ02BN_SAMPLES * 2
            );
            std::slice::from_raw_parts(ptr, len)
        }));
        let bluenoise_textures = device.create_bindless_array(N_BLUE_NOISE_TEXTURES);
        for i in 0..N_BLUE_NOISE_TEXTURES {
            let tex = device.create_tex2d::<f32>(
                luisa::PixelStorage::Short1,
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
                luisa::Sampler {
                    filter: luisa::SamplerFilter::Point,
                    address: luisa::SamplerAddress::Repeat,
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
    static ref PERMUTE_ELEMENT: Callable<(Expr<u32>, Expr<u32>, Expr<u32>, Expr<u32>), Expr<u32>> =
        create_static_callable::<(Expr<u32>, Expr<u32>, Expr<u32>, Expr<u32>), Expr<u32>>(
            |i: Expr<u32>, l: Expr<u32>, w: Expr<u32>, p: Expr<u32>| {
                let i = var!(u32, i);
                loop_!({
                    *i.get_mut() ^= p;
                    *i.get_mut() *= 0xe170893du32;
                    *i.get_mut() ^= p >> 16u32;
                    *i.get_mut() ^= (*i & w) >> 4u32;
                    *i.get_mut() ^= p >> 8u32;
                    *i.get_mut() *= 0x0929eb3fu32;
                    *i.get_mut() ^= p >> 23u32;
                    *i.get_mut() ^= (*i & w) >> 1u32;
                    *i.get_mut() *= 1 | p >> 27u32;
                    *i.get_mut() *= 0x6935fa69u32;
                    *i.get_mut() ^= (*i & w) >> 11u32;
                    *i.get_mut() *= 0x74dcb303u32;
                    *i.get_mut() ^= (*i & w) >> 2u32;
                    *i.get_mut() *= 0x9e501cc3u32;
                    *i.get_mut() ^= (*i & w) >> 2u32;
                    *i.get_mut() *= 0xc860a3dfu32;
                    *i.get_mut() &= w;
                    *i.get_mut() ^= *i >> 5u32;
                    if_!(i.cmplt(l), {
                        break_();
                    })
                });
                let i = *i;
                (i + p) % l
            }
        );
}
fn permute_element(i: Expr<u32>, l: Expr<u32>, w: Expr<u32>, p: Expr<u32>) -> Expr<u32> {
    PERMUTE_ELEMENT.call(i, l, w, p)
}
#[derive(Clone, Copy, Value, Debug)]
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
    pmj02bn_samples: Arc<Buffer<f32>>,
    pixel_samples: Arc<Buffer<Float2>>,
    bluenoise_textures: Arc<BindlessArray>,
    i: Expr<u32>,
    state: Var<Pmj02BnState>,
    states: Arc<Buffer<Pmj02BnState>>,
    forget: Var<bool>,
    next_1d: Arc<DynCallable<(Var<Pmj02BnState>,), Expr<f32>>>,
    next_2d: Arc<DynCallable<(Var<Pmj02BnState>,), Expr<Float2>>>,
}
impl Pmj02BnSampler {
    fn new(
        device: &Device,
        pmj02bn_samples: Arc<Buffer<f32>>,
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
                    .x()
            };
        let next_1d = {
            let bluenoise_textures = bluenoise_textures.clone();
            device.create_dyn_callable::<(Var<Pmj02BnState>,), Expr<f32>>(Box::new(
                move |state: Var<Pmj02BnState>| {
                    let hash = xxhash32_4(make_uint4(
                        *state.pixel().x(),
                        *state.pixel().y(),
                        *state.dim(),
                        *state.seed(),
                    ));
                    let index =
                        permute_element(*state.sample_index(), *state.spp(), *state.w(), hash);
                    let delta =
                        bluenoise(&bluenoise_textures, *state.sample_index(), *state.pixel());
                    *state.dim().get_mut() += 1;
                    ((index.float() + delta) / state.spp().float()).min(ONE_MINUS_EPSILON)
                },
            ))
        };
        let next_2d = {
            let pmj02bn_samples = pmj02bn_samples.clone();
            let bluenoise_textures = bluenoise_textures.clone();
            device.create_dyn_callable::<(Var<Pmj02BnState>,), Expr<Float2>>(Box::new(
                move |state: Var<Pmj02BnState>| {
                    let index = var!(u32, *state.sample_index());
                    let pmj_instance = *state.dim() / 2;
                    if_!(pmj_instance.cmpge(N_PMJ02BN_SETS as u32), {
                        let hash = xxhash32_4(make_uint4(
                            *state.pixel().x(),
                            *state.pixel().y(),
                            *state.dim(),
                            *state.seed(),
                        ));
                        *index.get_mut() =
                            permute_element(*state.sample_index(), *state.spp(), *state.w(), hash);
                    });
                    let u = pmj02bn_sample(&pmj02bn_samples, pmj_instance, *index);
                    let u =
                        u + bluenoise(&bluenoise_textures, *state.sample_index(), *state.pixel());
                    *state.dim().get_mut() += 2;
                    u.fract()
                },
            ))
        };
        Self {
            pmj02bn_samples,
            pixel_samples,
            bluenoise_textures,
            state: var!(Pmj02BnState, states.read(i)),
            forget: var!(bool, false),
            states,
            i,
            next_1d: Arc::new(next_1d),
            next_2d: Arc::new(next_2d),
        }
    }
}
impl Drop for Pmj02BnSampler {
    fn drop(&mut self) {
        if_!(!*self.forget, {
            let i = self.i;
            self.states.write(i, *self.state);
        });
    }
}
impl Sampler for Pmj02BnSampler {
    fn next_1d(&self) -> Float {
        (self.next_1d).call(self.state)
    }
    fn next_2d(&self) -> Expr<Float2> {
        (self.next_2d).call(self.state)
    }
    fn start(&self) {
        *self.state.dim().get_mut() = 0u32.into();
        if_!(
            self.state.sample_index().cmpeq(u32::MAX),
            {
                *self.state.sample_index().get_mut() = 0.into();
            },
            {
                *self.state.sample_index().get_mut() += 1;
            }
        );
    }
    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(Self {
            pmj02bn_samples: self.pmj02bn_samples.clone(),
            pixel_samples: self.pixel_samples.clone(),
            bluenoise_textures: self.bluenoise_textures.clone(),
            i: self.i,
            state: var!(Pmj02BnState, *self.state),
            states: self.states.clone(),
            forget: var!(bool, *self.forget),
            next_1d: self.next_1d.clone(),
            next_2d: self.next_2d.clone(),
        })
    }
    fn forget(&self) {
        *self.forget.get_mut() = true.into();
    }
}
impl SamplerCreator for Pmj02BnSamplerCreator {
    fn create(&self, pixel: Expr<Uint2>) -> Box<dyn Sampler> {
        let i = pixel.x() + pixel.y() * self.resolution.x;
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
