use crate::*;
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
}

pub struct LcgSampler {
    pub state: Var<u32>,
}
impl Sampler for LcgSampler {
    fn next_1d(&self) -> Float {
        const LCG_A: u32 = 1664525u32;
        const LCG_C: u32 = 1013904223u32;
        self.state.store(LCG_A * self.state.load() + LCG_C);
        self.state.load().float() / u32::MAX as f32
    }
}
