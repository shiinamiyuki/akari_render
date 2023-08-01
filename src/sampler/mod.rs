use lazy_static::lazy_static;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};

use crate::*;
pub mod mcmc;

pub trait Sampler {
    fn next_1d(&self) -> Float;
    fn next_2d(&self) -> Expr<Float2> {
        let u0 = self.next_1d();
        let u1 = self.next_1d();
        make_float2(u0, u1)
    }
    fn next_3d(&self) -> Expr<Float3> {
        let u0 = self.next_1d();
        let u1 = self.next_1d();
        let u3 = self.next_1d();
        make_float3(u0, u1, u3)
    }
    fn next_4d(&self) -> Expr<Float4> {
        let u0 = self.next_1d();
        let u1 = self.next_1d();
        let u3 = self.next_1d();
        let u4 = self.next_1d();
        make_float4(u0, u1, u3, u4)
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

pub struct PathSampler<S: Sampler> {
    base: S,
    bsdf_cnt: Var<u32>,
    light_cnt: Var<u32>,
    roulette_cnt: Var<u32>,
    bounces: u32,
    pixel: Var<Float2>,
    camera: ArrayVar<f32, 5>,
    light: VLArrayVar<f32>,
    bsdf: VLArrayVar<f32>,
    roulette: VLArrayVar<f32>,
}
impl<S: Sampler> PathSampler<S> {
    pub fn new(base: S, bounces: u32) -> Self {
        Self {
            base,
            bsdf_cnt: var!(u32),
            light_cnt: var!(u32),
            roulette_cnt: var!(u32),
            bounces,
            light: VLArrayVar::zero(bounces as usize * SampleStream::Light.dimension() as usize),
            bsdf: VLArrayVar::zero(bounces as usize * SampleStream::Bsdf.dimension() as usize),
            roulette: VLArrayVar::zero(bounces as usize),
            pixel: var!(Float2),
            camera: var!([f32; 5]),
        }
    }
    pub fn start(&self) {
        self.bsdf_cnt.store(0);
        self.light_cnt.store(0);
        self.roulette_cnt.store(0);
        self.base.start();
        for_range(const_(0)..const_(self.bounces as i32), |_| {});
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
#[derive(Clone)]
pub struct IndependentSampler {
    pub state: Pcg32Var,
}
impl Sampler for IndependentSampler {
    fn next_1d(&self) -> Float {
        let n = self.state.gen_u32();
        n.float() * ((1.0 / u32::MAX as f64) as f32)
    }
    fn start(&self) {}
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
}
