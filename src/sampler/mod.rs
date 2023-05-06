use crate::*;
pub mod mcmc;
pub trait IndependentSampler: Sampler {}
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
#[derive(Clone)]
pub struct LcgSampler {
    pub state: Var<u32>,
}
impl IndependentSampler for LcgSampler {}
impl Sampler for LcgSampler {
    fn next_1d(&self) -> Float {
        const LCG_A: u32 = 1664525u32;
        const LCG_C: u32 = 1013904223u32;
        self.state.store(LCG_A * self.state.load() + LCG_C);
        self.state.load().float() / u32::MAX as f32
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
}
pub struct IndependentReplaySampler<S: IndependentSampler> {
    pub base: S,
    pub sample: PrimarySample,
    pub cur_dim: Var<u32>,
}
impl<S: IndependentSampler> IndependentReplaySampler<S> {
    pub fn new(base: S, sample: PrimarySample) -> Self {
        Self {
            base,
            sample,
            cur_dim: var!(u32, 0),
        }
    }
}
impl<S: IndependentSampler> Sampler for IndependentReplaySampler<S> {
    fn next_1d(&self) -> Float {
        if_!(self.cur_dim.load().cmplt(self.sample.values.len()), {
            let ret = self.sample.values.read(self.cur_dim.load());
            self.cur_dim.store(self.cur_dim.load() + 1);
            ret
        }, else {
            self.base.next_1d()
        })
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
}
