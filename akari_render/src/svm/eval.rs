use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;

use super::texture::{rgb_gamma_correction, rgb_to_target_colorspace, spectral_uplift};
use super::{CompiledShader, ShaderRef, Svm, SvmNodeRef};
use crate::color::{Color, ColorPipeline, ColorSpaceId, SampledWavelengths};
use crate::interaction::SurfaceInteraction;
use crate::svm::{surface::*, *};
#[derive(Clone, Copy)]
pub enum SvmEvalInput {
    Surface {
        si: Expr<SurfaceInteraction>,
        swl: Expr<SampledWavelengths>,
    },
    // Volume {},
}
pub struct SvmEvaluator<'a> {
    pub color_pipeline: ColorPipeline,
    svm: &'a Svm,
    shader: &'a CompiledShader,
    env: HashMap<SvmNodeRef, Box<dyn Any>>,
    shader_data: ByteBufferVar,
    shader_ref: Expr<ShaderRef>,
    input: SvmEvalInput,
}

impl<'a> SvmEvaluator<'a> {
    pub fn new(
        svm: &'a Svm,
        color_pipeline: ColorPipeline,
        shader: &'a CompiledShader,
        shader_ref: Expr<ShaderRef>,
        shader_data: ByteBufferVar,
        input: SvmEvalInput,
    ) -> Self {
        Self {
            svm,
            color_pipeline,
            shader,
            env: HashMap::new(),
            shader_data,
            shader_ref,
            input,
        }
    }
    pub fn color_repr(&self) -> ColorRepr {
        self.color_pipeline.color_repr
    }
    pub fn swl(&self) -> Expr<SampledWavelengths> {
        match &self.input {
            SvmEvalInput::Surface { swl, .. } => *swl,
        }
    }
    pub fn si(&self) -> Expr<SurfaceInteraction> {
        match &self.input {
            SvmEvalInput::Surface { si, .. } => *si,
        }
    }
    fn node_offset(&self, index: u32) -> u32 {
        self.shader.node_offset[index as usize] as u32
    }
    fn get_node_expr<T: Value>(&self, index: u32) -> Expr<T> {
        let offset = self.node_offset(index);
        self.shader_data
            .read::<T>(offset + self.shader_ref.offset())
    }
    fn do_eval(&mut self, node: SvmNodeRef) -> Box<dyn Any> {
        let idx = node.index as usize;
        let node = &self.shader.nodes[idx];
        match node {
            svm::SvmNode::Float(f) => {
                let value = self.get_node_expr::<SvmFloat>(idx as u32).value();
                Box::new(value)
            }
            svm::SvmNode::Float3(f3) => {
                let value = self.get_node_expr::<SvmFloat3>(idx as u32).value().unpack();
                Box::new(value)
            }
            svm::SvmNode::MakeFloat3(mk_f3) => {
                let x = self.eval_float(mk_f3.x);
                let y = self.eval_float(mk_f3.y);
                let z = self.eval_float(mk_f3.z);
                Box::new(make_float3(x, y, z))
            }
            svm::SvmNode::RgbTex(rgb_tex) => {
                let rgb = self.eval_float3(rgb_tex.rgb);
                let colorspace = ColorSpaceId::to_colorspace(rgb_tex.colorspace);
                Box::new(rgb_to_target_colorspace(
                    rgb,
                    colorspace,
                    self.color_pipeline.rgb_colorspace,
                ))
            }
            svm::SvmNode::RgbImageTex(img_tex) => {
                let tex_idx = self.get_node_expr::<SvmRgbImageTex>(idx as u32).tex_idx();
                let textures = &self.svm.image_textures.var();
                let texture = textures.tex2d(tex_idx);
                let uv = self.si().geometry().uv();
                let rgb = texture.sample(uv.fract()).xyz();
                let colorspace = ColorSpaceId::to_colorspace(img_tex.colorspace);
                Box::new(rgb_gamma_correction(rgb, colorspace))
            }
            svm::SvmNode::SpectralUplift(uplift) => {
                let rgb = self.eval_float3(uplift.rgb);
                Box::new(spectral_uplift(
                    rgb,
                    self.color_pipeline.rgb_colorspace,
                    self.swl(),
                    self.color_repr(),
                ))
            }
            svm::SvmNode::DiffuseBsdf(bsdf) => Box::new(bsdf.closure(self)),
            svm::SvmNode::PrincipledBsdf(bsdf) => Box::new(bsdf.closure(self)),
            svm::SvmNode::MaterialOutput(out) => {
                let closure = self.eval_bsdf_closure(out.surface);
                Box::new(closure.clone())
            }
        }
    }
    fn eval(&self, node: SvmNodeRef) -> &dyn Any {
        &self.env[&node]
    }
    pub fn eval_float(&self, node: SvmNodeRef) -> Expr<f32> {
        self.eval(node)
            .downcast_ref::<Expr<f32>>()
            .copied()
            .unwrap()
    }
    pub fn eval_float3(&self, node: SvmNodeRef) -> Expr<Float3> {
        self.eval(node)
            .downcast_ref::<Expr<Float3>>()
            .copied()
            .unwrap()
    }
    pub fn eval_float4(&self, node: SvmNodeRef) -> Expr<Float4> {
        self.eval(node)
            .downcast_ref::<Expr<Float4>>()
            .copied()
            .unwrap()
    }
    pub fn eval_color(&self, node: SvmNodeRef) -> Color {
        self.eval(node).downcast_ref::<Color>().copied().unwrap()
    }
    pub fn eval_bsdf_closure(&self, node: SvmNodeRef) -> Rc<dyn Surface> {
        self.eval(node)
            .downcast_ref::<Rc<dyn Surface>>()
            .cloned()
            .unwrap()
    }
    pub fn eval_shader(mut self) -> Box<dyn Any> {
        let nodes = &self.shader.nodes;
        for i in 0..nodes.len() {
            self.do_eval(SvmNodeRef { index: i as u32 });
        }
        let last_idx = nodes.len() - 1;
        self.env
            .remove(&SvmNodeRef {
                index: last_idx as u32,
            })
            .unwrap()
    }
}
impl Svm {
    pub fn dispatch_surface<R: Aggregate>(
        &self,
        shader_ref: Expr<ShaderRef>,
        color_pipeline: ColorPipeline,
        si: Expr<SurfaceInteraction>,
        swl: Expr<SampledWavelengths>,
        f: impl Fn(&SurfaceClosure) -> R,
    ) -> R {
        let input = SvmEvalInput::Surface { si, swl };
        let mut kinds = self
            .surface_shaders
            .shaders
            .keys()
            .copied()
            .collect::<Vec<_>>();
        kinds.sort();
        let mut sw = switch::<R>(shader_ref.shader_kind().int());
        for k in kinds {
            sw = sw.case(k as i32, || {
                let eval = SvmEvaluator::new(
                    self,
                    color_pipeline,
                    &self.surface_shaders.shaders[&k],
                    shader_ref,
                    self.surface_shaders.shader_data.var(),
                    input,
                );
                let bsdf = eval
                    .eval_shader()
                    .downcast_ref::<Rc<dyn Surface>>()
                    .unwrap()
                    .clone();
                let closure = SurfaceClosure {
                    inner: bsdf,
                    frame: si.frame(),
                };
                f(&closure)
            });
        }
        sw.finish()
    }
}
