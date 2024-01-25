use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::texture::{rgb_gamma_correction, rgb_to_target_colorspace, spectral_uplift};
use super::{CompiledShader, ShaderRef, Svm, SvmNodeRef};
use crate::color::{Color, ColorPipeline, ColorSpaceId, SampledWavelengths};
use crate::geometry::{Frame, FrameComps, FrameExpr};
use crate::interaction::SurfaceInteraction;
use crate::svm::{surface::*, *};
#[derive(Clone, Copy)]
pub enum SvmEvalInput {
    SurfaceAlpha {
        si: SurfaceInteraction,
    },
    Surface {
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
    },
    // Volume {},
}
struct Evaluated {
    value: Box<dyn Any>,
    field: Box<dyn Fn(&dyn Any, String) -> Box<dyn Any>>,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum SvmEvalMode {
    /// only evaluate the alpha
    Alpha,
    /// evalutes full surface closure
    Surface,
}
pub struct SvmEvaluator<'a> {
    pub color_pipeline: ColorPipeline,
    pub svm: &'a Svm,
    shader: &'a ShaderBytecode,
    env: HashMap<SvmNodeRef, Evaluated>,
    shader_data: ByteBufferVar,
    shader_ref: Expr<ShaderRef>,
    input: SvmEvalInput,
    mode: SvmEvalMode,
}

#[derive(Clone, Copy)]
struct ColorAlpha {
    color: Color,
    alpha: Expr<f32>,
}

impl<'a> SvmEvaluator<'a> {
    pub fn new(
        svm: &'a Svm,
        color_pipeline: ColorPipeline,
        shader: &'a ShaderBytecode,
        shader_ref: Expr<ShaderRef>,
        shader_data: ByteBufferVar,
        input: SvmEvalInput,
        mode: SvmEvalMode,
    ) -> Self {
        Self {
            svm,
            color_pipeline,
            shader,
            env: HashMap::new(),
            shader_data,
            shader_ref,
            input,
            mode,
        }
    }
    pub fn mode(&self) -> SvmEvalMode {
        self.mode
    }
    pub fn color_repr(&self) -> ColorRepr {
        self.color_pipeline.color_repr
    }
    pub fn swl(&self) -> Expr<SampledWavelengths> {
        match &self.input {
            SvmEvalInput::Surface { swl, .. } => *swl,
            SvmEvalInput::SurfaceAlpha { si } => **SampledWavelengths::var_zeroed(),
        }
    }
    pub fn si(&self) -> SurfaceInteraction {
        match &self.input {
            SvmEvalInput::Surface { si, .. } => *si,
            SvmEvalInput::SurfaceAlpha { si } => *si,
        }
    }

    #[tracked(crate = "luisa")]
    fn read_data<T: Value>(&self, cst: SvmConst<T>) -> Expr<T> {
        self.shader_data
            .read_as::<T>(cst.offset + self.shader_ref.data_offset)
    }
    #[tracked(crate = "luisa")]
    fn do_eval(&mut self, node: SvmNodeRef) -> Evaluated {
        let idx = node.index as usize;
        let node = &self.shader.nodes[idx];
        macro_rules! wrap_any {
            ($v:expr) => {{
                add_type_name(&$v);
                let value = Box::new($v) as Box<dyn Any>;
                Evaluated {
                    value,
                    field: Box::new(|_, s| panic!("Field {} not found", s)),
                }
            }};
        }
        match node {
            svm::SvmNode::Float(v) => {
                let value = self.read_data(*v);
                wrap_any!(value)
            }
            svm::SvmNode::Float3(v) => {
                let value = self.read_data(*v);
                wrap_any!(value)
            }
            svm::SvmNode::MakeFloat3(mk_f3) => {
                let x = self.eval_float(mk_f3.x);
                let y = self.eval_float(mk_f3.y);
                let z = self.eval_float(mk_f3.z);
                wrap_any!(Float3::expr(x, y, z))
            }
            svm::SvmNode::RgbTex(rgb_tex) => {
                let rgb = self.eval_float3(rgb_tex.rgb);

                let colorspace = ColorSpaceId::to_colorspace(rgb_tex.colorspace);
                wrap_any!(rgb_to_target_colorspace(
                    rgb,
                    colorspace,
                    self.color_pipeline.rgb_colorspace,
                )
                .extend(1.0))
            }
            svm::SvmNode::RgbImageTex(img_tex) => {
                let tex_idx = self.read_data(img_tex.tex_idx);

                let heap = &self.svm.heap.var();
                let texture = heap.tex2d(tex_idx);
                let uv = img_tex
                    .uv
                    .map(|uv| self.eval_float2_auto_convert(uv))
                    .unwrap_or_else(|| self.si().uv);
                let rgba = texture.sample(uv);
                let rgb = rgba.xyz();
                let alpha = rgba.w;

                let rgb = if img_tex.colorspace != ColorSpaceId::NONE {
                    let colorspace = ColorSpaceId::to_colorspace(img_tex.colorspace);
                    rgb_gamma_correction(rgb, colorspace)
                } else {
                    rgb
                };
                wrap_any!(rgb.extend(alpha))
            }
            svm::SvmNode::SpectralUplift(uplift) => {
                if self.mode == SvmEvalMode::Alpha {
                    let rgba = self.eval_float4(uplift.rgb);
                    wrap_any!(ColorAlpha {
                        color: Color::from_flat(self.color_repr(), rgba.xyz().extend(0.0)),
                        alpha: rgba.w,
                    })
                } else {
                    let rgba = self.eval_float4(uplift.rgb);
                    // cpu_dbg!(rgb);
                    wrap_any!(ColorAlpha {
                        color: spectral_uplift(
                            rgba.xyz(),
                            self.color_pipeline.rgb_colorspace,
                            self.swl(),
                            self.color_repr(),
                        ),
                        alpha: rgba.w,
                    })
                }
            }
            svm::SvmNode::Emission(bsdf) => wrap_any!(bsdf.closure(self)),
            svm::SvmNode::DiffuseBsdf(bsdf) => wrap_any!(bsdf.closure(self)),
            svm::SvmNode::PlasticBsdf(bsdf) => wrap_any!(bsdf.closure(self)),
            svm::SvmNode::MetalBsdf(bsdf) => wrap_any!(bsdf.closure(self)),
            svm::SvmNode::GlassBsdf(bsdf) => wrap_any!(bsdf.closure(self)),
            svm::SvmNode::PrincipledBsdf(bsdf) => wrap_any!(bsdf.closure(self)),
            svm::SvmNode::MaterialOutput(out) => {
                let closure = self.eval_bsdf_closure(out.surface);
                wrap_any!(closure.clone())
            }
            svm::SvmNode::NormalMap(nm) => {
                let normal = 2.0 * self.eval_float3_auto_convert(nm.normal) - 1.0;
                let strength = self.eval_float_auto_convert(nm.strength);
                let normal = if strength != 1.0 {
                    normal * Float3::expr(strength, strength, 1.0)
                } else {
                    normal
                };
                assert_eq!(
                    nm.space,
                    NormalMapSpace::TangentSpace,
                    "Only tangent space normal map is supported"
                );
                wrap_any!(normal)
            }
            svm::SvmNode::Mapping(m) => {
                let v = self.eval_float3_auto_convert(m.vector);
                let location = self.eval_float3_auto_convert(m.location);
                let scale = self.eval_float3_auto_convert(m.scale);
                // let rotation = self.eval_float3_auto_convert(m.rotation);
                // todo: rotation
                match m.ty {
                    MappingType::Point => {
                        let v = v * scale + location;
                        wrap_any!(v)
                    }
                    MappingType::Texture => {
                        let v = (v - location) / scale;
                        wrap_any!(v)
                    }
                }
            }
            svm::SvmNode::ExtractField(ef) => {
                let evaled: &Evaluated = &self
                    .env
                    .get(&ef.node)
                    .unwrap_or_else(|| panic!("Node {:?} not evaluated", ef.node));
                let value = (evaled.field)(evaled.value.as_ref(), ef.field.clone());
                Evaluated {
                    value,
                    field: Box::new(|_, s| panic!("Field {} not found", s)),
                }
            }
            svm::SvmNode::TexCoords(_) => Evaluated {
                value: Box::new(self.si().uv) as Box<dyn Any>,
                field: Box::new(|v, s| {
                    let uv = v.downcast_ref::<Expr<Float2>>().unwrap();
                    assert_eq!(s, "uv");
                    Box::new(uv.clone()) as Box<dyn Any>
                }),
            },
            svm::SvmNode::CheckerBoard(cb) => {
                let v = cb.vector.map(|v| self.eval_float2_auto_convert(v));
                let uv = v.unwrap_or_else(|| self.si().uv);
                let (color1, alpha1) = self.eval_color_alpha(cb.color1);
                let (color2, alpha2) = self.eval_color_alpha(cb.color2);
                let scale = self.eval_float(cb.scale);
                let pos = (uv * scale * 2.0).floor().cast_i32();
                let (color, alpha) = if (pos.x + pos.y) % 2 == 0 {
                    (color1, alpha1)
                } else {
                    (color2, alpha2)
                };
                wrap_any!(ColorAlpha { color, alpha })
            }
            svm::SvmNode::SeparateColor(sc) => {
                let color = self.eval_float3_auto_convert(sc.color);
                Evaluated {
                    value: Box::new(()),
                    field: Box::new(move |_, s| {
                        let f = match s.as_ref() {
                            "Red" => color.x,
                            "Green" => color.y,
                            "Blue" => color.z,
                            _ => panic!("Field {} not found", s),
                        };
                        Box::new(f) as Box<dyn Any>
                    }),
                }
            }
        }
    }
    fn eval<T: Any + Clone>(&self, node: SvmNodeRef) -> T {
        let any: &Box<dyn Any> = &self
            .env
            .get(&node)
            .unwrap_or_else(|| panic!("Node {:?} not evaluated", node))
            .value;
        let any = any.as_ref();
        let type_id = std::any::TypeId::of::<T>();
        if any.type_id() != type_id {
            panic!(
                "Node {:?} evaluated as {:?}, expected {:?}",
                node,
                type_id_to_name(any.type_id()),
                type_id_to_name(type_id)
            );
        }
        any.downcast_ref::<T>().cloned().unwrap()
    }
    fn try_eval<T: Any + Clone>(&self, node: SvmNodeRef) -> Option<T> {
        let any: &Box<dyn Any> = &self
            .env
            .get(&node)
            .unwrap_or_else(|| panic!("Node {:?} not evaluated", node))
            .value;
        let any = any.as_ref();
        any.downcast_ref::<T>().cloned()
    }
    pub fn eval_float(&self, node: SvmNodeRef) -> Expr<f32> {
        self.eval(node)
    }
    pub fn eval_float3(&self, node: SvmNodeRef) -> Expr<Float3> {
        self.eval(node)
    }
    pub fn eval_float2_auto_convert(&self, node: SvmNodeRef) -> Expr<Float2> {
        let f2 = self.try_eval::<Expr<Float2>>(node);
        if let Some(f2) = f2 {
            return f2;
        }
        let f3 = self.try_eval::<Expr<Float3>>(node);
        if let Some(f3) = f3 {
            return f3.xy();
        }
        let f4 = self.try_eval::<Expr<Float4>>(node);
        if let Some(f4) = f4 {
            return f4.xy();
        }
        let f = self.eval_float(node);
        Float2::expr(f, 0.0)
    }
    pub fn eval_float3_auto_convert(&self, node: SvmNodeRef) -> Expr<Float3> {
        let f3 = self.try_eval::<Expr<Float3>>(node);
        if let Some(f3) = f3 {
            return f3;
        }
        let f2 = self.try_eval::<Expr<Float2>>(node);
        if let Some(f2) = f2 {
            return f2.extend(0.0);
        }
        let f4 = self.try_eval::<Expr<Float4>>(node);
        if let Some(f4) = f4 {
            return f4.xyz();
        }
        let f = self.eval_float(node);
        Float3::expr(f, 0.0, 0.0)
    }
    pub fn eval_float_auto_convert(&self, node: SvmNodeRef) -> Expr<f32> {
        let f = self.try_eval::<Expr<f32>>(node);
        if let Some(f) = f {
            return f;
        }
        let f2 = self.try_eval::<Expr<Float2>>(node);
        if let Some(f2) = f2 {
            return f2.x;
        }
        let f3 = self.try_eval::<Expr<Float3>>(node);
        if let Some(f3) = f3 {
            return f3.x;
        }
        let f4 = self.eval::<Expr<Float4>>(node);
        f4.x
    }
    pub fn eval_float4(&self, node: SvmNodeRef) -> Expr<Float4> {
        self.eval(node)
    }
    pub fn eval_color(&self, node: SvmNodeRef) -> Color {
        self.eval_color_alpha(node).0
    }
    pub fn eval_color_alpha(&self, node: SvmNodeRef) -> (Color, Expr<f32>) {
        let c: ColorAlpha = self.eval(node);
        (c.color, c.alpha)
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
            .value
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
                SvmEvalMode::Surface,
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

    pub fn dispatch_svm<R: Aggregate>(
        &self,
        shader_ref: Expr<ShaderRef>,
        color_pipeline: ColorPipeline,
        si: SurfaceInteraction,
        swl: Option<Expr<SampledWavelengths>>,
        mode: SvmEvalMode,
        f: impl Fn(SvmEvaluator) -> R,
    ) -> R {
        let input = match mode {
            SvmEvalMode::Alpha => SvmEvalInput::SurfaceAlpha { si },
            SvmEvalMode::Surface => SvmEvalInput::Surface {
                si,
                swl: swl.unwrap(),
            },
        };
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
                    mode,
                );
                f(eval)
            });
        }
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
        self.dispatch_svm(
            shader_ref,
            color_pipeline,
            si,
            Some(swl),
            SvmEvalMode::Surface,
            |eval| {
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
            },
        )
        // let input = SvmEvalInput::Surface { si, swl };
        // let mut kinds = self
        //     .surface_shaders
        //     .kind_to_shader
        //     .keys()
        //     .copied()
        //     .collect::<Vec<_>>();
        // kinds.sort();
        // let mut sw = switch::<R>(shader_ref.shader_kind.cast_i32());
        // for k in kinds {
        //     sw = sw.case(k as i32, || {
        //         let eval = SvmEvaluator::new(
        //             self,
        //             color_pipeline,
        //             &self.surface_shaders.kind_to_shader[&k],
        //             shader_ref,
        //             self.surface_shaders.shader_data.var(),
        //             input,
        //         );
        //         let bsdf = eval
        //             .eval_shader()
        //             .downcast_ref::<Rc<dyn Surface>>()
        //             .unwrap()
        //             .clone();
        //         let closure = SurfaceClosure {
        //             inner: bsdf,
        //             frame: si.frame,
        //             ng: si.ng,
        //         };
        //         f(&closure)
        //     });
        // }
        // sw.finish()
    }
}

thread_local! {
    static TYPE_ID_TO_NAME: RefCell<HashMap<std::any::TypeId, &'static str>> = RefCell::new(HashMap::new());
}
fn add_type_name<T: 'static>(x: &T) {
    TYPE_ID_TO_NAME.with(|map| {
        let mut map = map.borrow_mut();
        let type_id = std::any::TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();
        map.insert(type_id, type_name);
    });
}

fn type_id_to_name(type_id: std::any::TypeId) -> String {
    TYPE_ID_TO_NAME.with(|map| {
        let map = map.borrow();
        map.get(&type_id)
            .map(|x| x.to_string())
            .unwrap_or(format!("{:?}", type_id))
    })
}
