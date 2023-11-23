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
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
    },
    // Volume {},
}
pub struct SvmEvaluator<'a> {
    pub color_pipeline: ColorPipeline,
    svm: &'a Svm,
    shader: &'a ShaderBytecode,
    env: HashMap<SvmNodeRef, Box<dyn Any>>,
    shader_data: ByteBufferVar,
    shader_ref: Expr<ShaderRef>,
    input: SvmEvalInput,
}

impl<'a> SvmEvaluator<'a> {
    pub fn new(
        svm: &'a Svm,
        color_pipeline: ColorPipeline,
        shader: &'a ShaderBytecode,
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
    pub fn si(&self) -> SurfaceInteraction {
        match &self.input {
            SvmEvalInput::Surface { si, .. } => *si,
        }
    }

    #[tracked(crate = "luisa")]
    fn read_data<T: Value>(&self, cst: SvmConst<T>) -> Expr<T> {
        unsafe {
            self.shader_data
                .read_as::<T>(cst.offset + self.shader_ref.data_offset)
        }
    }
    #[tracked(crate = "luisa")]
    fn do_eval(&mut self, node: SvmNodeRef) -> Box<dyn Any> {
        let idx = node.index as usize;
        let node = &self.shader.nodes[idx];
        match node {
            svm::SvmNode::Float(v) => {
                let value = self.read_data(*v);
                Box::new(value)
            }
            svm::SvmNode::Float3(v) => {
                let value = self.read_data(*v);
                Box::new(value)
            }
            svm::SvmNode::MakeFloat3(mk_f3) => {
                let x = self.eval_float(mk_f3.x);
                let y = self.eval_float(mk_f3.y);
                let z = self.eval_float(mk_f3.z);
                Box::new(Float3::expr(x, y, z))
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
                let tex_idx = self.read_data(img_tex.tex_idx);

                let heap = &self.svm.heap.var();
                let texture = heap.tex2d(tex_idx);
                let uv = self.si().uv;
                let rgb = texture.sample(uv).xyz();
                let colorspace = ColorSpaceId::to_colorspace(img_tex.colorspace);
                let rgb = rgb_gamma_correction(rgb, colorspace);
                if debug_mode() {
                    lc_assert!(rgb.reduce_min().ge(0.0));
                }
                Box::new(rgb)
            }
            svm::SvmNode::SpectralUplift(uplift) => {
                let rgb = self.eval_float3(uplift.rgb);
                // cpu_dbg!(rgb);
                Box::new(spectral_uplift(
                    rgb,
                    self.color_pipeline.rgb_colorspace,
                    self.swl(),
                    self.color_repr(),
                ))
            }
            svm::SvmNode::Emission(bsdf) => Box::new(bsdf.closure(self)),
            svm::SvmNode::DiffuseBsdf(bsdf) => Box::new(bsdf.closure(self)),
            svm::SvmNode::GlassBsdf(bsdf) => Box::new(bsdf.closure(self)),
            svm::SvmNode::PrincipledBsdf(bsdf) => Box::new(bsdf.closure(self)),
            svm::SvmNode::MaterialOutput(out) => {
                let closure = self.eval_bsdf_closure(out.surface);
                Box::new(closure.clone())
            }
        }
    }
    fn eval<T: Any + Clone>(&self, node: SvmNodeRef) -> T {
        let any = self
            .env
            .get(&node)
            .unwrap_or_else(|| panic!("Node {:?} not evaluated", node));
        let any = any.as_ref();
        let type_id = std::any::TypeId::of::<T>();
        if any.type_id() != type_id {
            panic!(
                "Node {:?} evaluated as {:?}, expected {:?}",
                node,
                any.type_id(),
                type_id
            );
        }
        any.downcast_ref::<T>().cloned().unwrap()
    }
    pub fn eval_float(&self, node: SvmNodeRef) -> Expr<f32> {
        self.eval(node)
    }
    pub fn eval_float3(&self, node: SvmNodeRef) -> Expr<Float3> {
        self.eval(node)
    }
    pub fn eval_float4(&self, node: SvmNodeRef) -> Expr<Float4> {
        self.eval(node)
    }
    pub fn eval_color(&self, node: SvmNodeRef) -> Color {
        self.eval(node)
    }
    pub fn eval_bsdf_closure(&self, node: SvmNodeRef) -> Rc<dyn Surface> {
        self.eval(node)
    }
    pub fn eval_shader(mut self) -> Box<dyn Any> {
        let nodes = &self.shader.nodes;
        // dbg!(nodes);
        for i in 0..nodes.len() {
            let v = self.do_eval(SvmNodeRef { index: i as u32 });
            let old = self.env.insert(SvmNodeRef { index: i as u32 }, v);
            assert!(old.is_none());
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
    pub fn dispatch_surface_single_kind<R: Aggregate>(
        &self,
        kind: u32,
        shader_ref: Expr<ShaderRef>,
        color_pipeline: ColorPipeline,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        f: impl Fn(&SurfaceClosure) -> R,
    ) -> R {
        let input = SvmEvalInput::Surface { si, swl };
        let mut kinds = self
            .surface_shaders
            .kind_to_shader
            .keys()
            .copied()
            .collect::<Vec<_>>();
        kinds.sort();
        let mut sw = switch::<R>(shader_ref.shader_kind.cast_i32());
        if debug_mode() {
            lc_assert!(kind.eq(shader_ref.shader_kind));
        }
        sw = sw.case(kind as i32, || {
            let eval = SvmEvaluator::new(
                self,
                color_pipeline,
                &self.surface_shaders.kind_to_shader[&kind],
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
                frame: si.frame,
                ng: si.ng,
            };
            f(&closure)
        });
        sw.finish()
    }

    pub fn dispatch_surface<R: Aggregate>(
        &self,
        shader_ref: Expr<ShaderRef>,
        color_pipeline: ColorPipeline,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        f: impl Fn(&SurfaceClosure) -> R,
    ) -> R {
        let input = SvmEvalInput::Surface { si, swl };
        let mut kinds = self
            .surface_shaders
            .kind_to_shader
            .keys()
            .copied()
            .collect::<Vec<_>>();
        kinds.sort();
        let mut sw = switch::<R>(shader_ref.shader_kind.cast_i32());
        for k in kinds {
            sw = sw.case(k as i32, || {
                let eval = SvmEvaluator::new(
                    self,
                    color_pipeline,
                    &self.surface_shaders.kind_to_shader[&k],
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
                    frame: si.frame,
                    ng: si.ng,
                };
                f(&closure)
            });
        }
        sw.finish()
    }
}
