use rand::{thread_rng, Rng};

use crate::*;
pub mod mcmc;

pub trait Sampler {
    fn next_1d(&self) -> Float;
    fn next_2d(&self) -> Expr<Float2> {
        make_float2(self.next_1d(), self.next_1d())
    }
    fn next_3d(&self) -> Expr<Float3> {
        make_float3(self.next_1d(), self.next_1d(), self.next_1d())
    }
    fn next_4d(&self) -> Expr<Float4> {
        make_float4(
            self.next_1d(),
            self.next_1d(),
            self.next_1d(),
            self.next_1d(),
        )
    }
    fn start(&self);
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
        let old_state = self.state().load();
        self.state()
            .store(old_state * Pcg32::PCG32_MULT + self.inc().load());
        let xor_shifted: Expr<u32> = (((old_state >> 18) ^ old_state) >> 27).uint();
        let rot = (old_state >> 59).uint();
        (xor_shifted >> rot) | (xor_shifted << ((!rot + 1) & 31))
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
            lc_assert!(false);
            self.base.next_1d()
        })
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
}
