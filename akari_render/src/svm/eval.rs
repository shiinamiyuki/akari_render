use crate::{
    color::Color,
    surface::{Bsdf, BsdfClosure},
    *,
};

use super::{CompiledShader, ShaderRef, Svm, SvmNodeRef};
pub struct SvmEvaluator<'a> {
    pub color_repr: ColorRepr,
    svm: &'a Svm,
    shader_kind: u32,
    shader: &'a CompiledShader,
}

impl<'a> SvmEvaluator<'a> {
    pub fn new(svm: &'a Svm, color_repr: ColorRepr, shader_kind: u32) -> Self {
        Self {
            svm,
            color_repr,
            shader_kind,
            shader: &svm.shaders[&shader_kind],
        }
    }
    pub fn eval_float(&self, node: Expr<SvmNodeRef>) -> Expr<f32> {
        todo!()
    }
    pub fn eval_float3(&self, node: Expr<SvmNodeRef>) -> Expr<Float3> {
        todo!()
    }
    pub fn eval_float4(&self, node: Expr<SvmNodeRef>) -> Expr<Float4> {
        todo!()
    }
    pub fn eval_color(&self, node: Expr<SvmNodeRef>) -> Color {
        todo!()
    }
    pub fn eval_bsdf_closure(&self, node: Expr<SvmNodeRef>) -> Box<dyn Bsdf> {
        todo!()
    }
}
impl Svm {
    pub fn dispatch_bsdf<R: Aggregate>(&self, f: impl FnOnce(&BsdfClosure) -> R) -> R {
        todo!()
    }
    pub fn eval_float_shader(&self, shader: Expr<ShaderRef>) {}
}
