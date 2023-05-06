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

// pub struct MetroplisSampler<S: IndependentSampler> {
//     pub base: S,
//     pub buffer: BufferVar<PrimarySample>,
//     pub start: Expr<u32>,
//     pub count: Expr<u32>,
//     pub cur_dim: Var<u32>,
//     pub large_step_prob: Expr<f32>,
//     pub large_step: Var<bool>,
// }
// impl<S: IndependentSampler> MetroplisSampler<S> {
//     pub fn new(
//         base: S,
//         buffer: BufferVar<PrimarySample>,
//         start: Expr<u32>,
//         count: Expr<u32>,
//         large_step_prob: Expr<f32>,
//     ) -> Self {
//         Self {
//             base,
//             buffer,
//             start,
//             count,
//             cur_dim: var!(u32, 0),
//             large_step_prob,
//             large_step: var!(bool, false),
//         }
//     }

//     pub fn reject(&self) {
//         for_range(const_(0)..self.cur_dim.load().int(), |i| {
//             let i = i.uint();
//             let s = self.buffer.read(self.start + i);
//             let s = s.set_value(s.backup());
//             self.buffer.write(self.start + i, s);
//         });
//     }
// }
// impl<S: IndependentSampler> Sampler for MetroplisSampler<S> {
//     fn next_1d(&self) -> Float {
//         if_!(self.cur_dim.load().cmplt(self.count), {
//             let ret = self.buffer.read(self.start + self.cur_dim.load()).value();
//             self.cur_dim.store(self.cur_dim.load() + 1);
//             ret
//         }, else {
//             self.base.next_1d()
//         })
//     }
//     fn start(&self) {
//         self.cur_dim.store(0);
//         self.large_step
//             .store(self.base.next_1d().cmplt(self.large_step_prob));
//         for_range(const_(0)..self.count.int(), |i| {
//             let i = i.uint();
//             let mut s = self.buffer.read(self.start + i);
//             s = s.set_backup(s.value());
//             self.buffer.write(self.start + i, s);
//         });
//     }
// }
