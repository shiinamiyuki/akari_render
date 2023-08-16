use std::{sync::Arc, time::Instant};

use luisa::rtx::offset_ray_origin;
use rand::Rng;

use super::{Integrator, RenderOptions};
use crate::{
    color::*,
    film::*,
    geometry::*,
    interaction::SurfaceInteraction,
    sampler::*,
    scene::*,
    surface::{Bsdf, Surface},
    *,
};
use serde::{Deserialize, Serialize};
#[derive(Clone)]
pub struct PathTracer {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub pixel_offset: Int2,
    pub seed: u64,
    config: Config,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub pixel_offset: [i32; 2],
    pub seed: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            spp: 256,
            max_depth: 7,
            rr_depth: 5,
            spp_per_pass: 64,
            use_nee: true,
            indirect_only: false,
            pixel_offset: [0, 0],
            seed: 0,
        }
    }
}
impl PathTracer {
    pub fn new(device: Device, config: Config) -> Self {
        Self {
            device,
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.indirect_only,
            pixel_offset: Int2::new(config.pixel_offset[0], config.pixel_offset[1]),
            seed: config.seed,
            config,
        }
    }
}
pub fn mis_weight(pdf_a: Expr<f32>, pdf_b: Expr<f32>, power: u32) -> Expr<f32> {
    let apply_power = |x: Expr<f32>| {
        let mut p = const_(1.0f32);
        for _ in 0..power {
            p = p * x;
        }
        p
    };
    let pdf_a = apply_power(pdf_a);
    let pdf_b = apply_power(pdf_b);
    pdf_a / (pdf_a + pdf_b)
}
pub struct VertexType;
impl VertexType {
    pub const INVALID: u32 = 0;
    pub const LAST_HIT_LIGHT: u32 = 1;
    pub const LAST_NEE: u32 = 2;
    pub const INTERIOR: u32 = 3;
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ReconnectionVertex {
    pub bary: Float2,
    pub direct: PackedFloat3,
    pub direct_wi: PackedFloat3,
    pub indirect: PackedFloat3,
    pub wo: PackedFloat3,
    pub wi: PackedFloat3,
    pub direct_light_pdf: f32,
    pub inst_id: u32,
    pub prim_id: u32,
    pub prev_bsdf_pdf: f32,
    pub bsdf_pdf: f32,
    pub dist: f32,
    pub depth: u32,
    pub type_: u32,
}
impl ReconnectionVertexVar {
    pub fn valid(&self) -> Expr<bool> {
        self.type_().load().cmpne(VertexType::INVALID)
    }
}
#[derive(Clone, Copy)]
pub struct ReconnectionShiftMapping {
    pub min_dist: Expr<f32>,
    pub is_base_path: Expr<bool>,
    pub vertex: Var<ReconnectionVertex>,
    pub jacobian: Var<f32>,
    pub success: Var<bool>,
    pub min_roughness: Expr<f32>,
}
#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
pub struct DenoiseFeatures {
    pub albedo: Float3,
    pub normal: Float3,
}
impl PathTracer {
    pub fn radiance(
        &self,
        scene: &Arc<Scene>,
        ray: Expr<Ray>,
        sampler: &dyn Sampler,
        eval: &Evaluators,
    ) -> Color {
        self.trace(scene, ray, sampler, eval, None, None)
    }
    pub fn trace(
        &self,
        scene: &Arc<Scene>,
        ray: Expr<Ray>,
        sampler: &dyn Sampler,
        eval: &Evaluators,
        features: Option<Var<DenoiseFeatures>>,
        shift_mapping: Option<ReconnectionShiftMapping>,
    ) -> Color {
        let color_repr = eval.color_repr;
        let l = ColorVar::zero(color_repr);
        let beta = ColorVar::one(color_repr);

        let reconnect_l = ColorVar::zero(color_repr);
        let reconnect_beta = ColorVar::zero(color_repr);
        let reconnection_vertex = shift_mapping.as_ref().map(|sm| sm.vertex);

        let ray = var!(Ray, ray);
        let depth = var!(u32, 0);
        let prev_bsdf_pdf = var!(f32);
        let prev_n = var!(Float3, ray.d());
        let prev_p = var!(Float3, ray.o());
        let prev_roughness = var!(f32, 0.0);
        let acc_radiance = |e: Color| {
            l.store(l.load() + beta.load() * e);
            if shift_mapping.is_some() {
                reconnect_l.store(reconnect_l.load() + reconnect_beta.load() * e);
            }
        };
        let acc_beta = |k: Color| {
            beta.store(beta.load() * k);
            if shift_mapping.is_some() {
                reconnect_beta.store(reconnect_beta.load() * k);
            }
        };
        if let Some(shift_mapping) = &shift_mapping {
            if_!(!shift_mapping.is_base_path, {
                shift_mapping.success.store(false);
            })
        }
        let found_reconnectible_vertex = var!(bool);
        const MIN_RECONNECT_DEPTH: u32 = 1;
        let debug_sm_failed_line = var!(u32, 0);
        loop_!({
            // if let Some(shift_mapping) = &shift_mapping {
            //     if_!(
            //         !shift_mapping.is_base_path & !shift_mapping.vertex.valid(),
            //         {
            //             // the base path does not have a valid reconnection vertex
            //             // if we proceed with random replay, the path is not invertible
            //             // so we return zero throughput
            //             break_();
            //         }
            //     )
            // }
            let si = scene.intersect(ray.load());
            let wo = -ray.d().load();
            if_!(
                si.valid(),
                {
                    let inst_id = si.inst_id();
                    let instance = scene.meshes.mesh_instances.var().read(inst_id);
                    if let Some(shift_mapping) = &shift_mapping {
                        let reconnection_vertex = reconnection_vertex.unwrap();
                        if_!(reconnection_vertex.valid() & shift_mapping.is_base_path, {
                            reconnection_vertex.set_type_(VertexType::INTERIOR);
                        });
                    }
                    if_!(
                        instance.light().valid() & (!self.indirect_only | depth.load().cmpgt(1)),
                        {
                            let direct = eval.light.le(ray.load(), si);
                            if_!(depth.load().cmpeq(0) | !self.use_nee, {
                                acc_radiance(direct);
                            }, else {
                                let pn = {
                                    let p = ray.o();
                                    let n = prev_n.load();
                                    PointNormalExpr::new(p, n)
                                };
                                let light_pdf = eval.light.pdf(si, pn);
                                let w = mis_weight(prev_bsdf_pdf.load(), light_pdf, 1);
                                acc_radiance(direct * w);
                            })
                        }
                    );
                    let p = si.geometry().p();
                    let ng = si.geometry().ng();
                    let surface = instance.surface();
                    if let Some(features) = features {
                        if_!(depth.load().cmpeq(0), {
                            features.set_normal(si.geometry().ns());
                            features.set_albedo(eval.bsdf.albedo(surface, si, wo).as_rgb());
                        });
                    }
                    depth.store(depth.load() + 1);
                    if let Some(shift_mapping) = &shift_mapping {
                        let reconnection_vertex = reconnection_vertex.unwrap();
                        let is_last_vertex = depth.load().cmpeq(self.max_depth);
                        let dist = (prev_p.load() - p).length();
                        let can_connect = dist.cmpgt(shift_mapping.min_dist)
                            & prev_roughness.load().cmpgt(shift_mapping.min_roughness);
                        if_!(
                            depth.load().cmpgt(MIN_RECONNECT_DEPTH + 1)
                                & can_connect
                                & is_last_vertex,
                            {
                                found_reconnectible_vertex
                                    .store(found_reconnectible_vertex.load() | can_connect);
                                if_!(
                                    !reconnection_vertex.valid() & shift_mapping.is_base_path,
                                    {
                                        reconnection_vertex.store(struct_!(ReconnectionVertex {
                                            direct: Float3Expr::zero().pack(), // should re-evaluate light::le
                                            direct_wi: Float3Expr::zero().pack(),
                                            direct_light_pdf: 0.0.into(),
                                            indirect: Float3Expr::zero().pack(),
                                            wo: wo.pack(),
                                            wi: Float3Expr::zero().pack(),
                                            inst_id: si.inst_id(),
                                            prim_id: si.prim_id(),
                                            bary: si.bary(),
                                            prev_bsdf_pdf: prev_bsdf_pdf.load(),
                                            bsdf_pdf: const_(0.0f32),
                                            dist: dist,
                                            depth: depth.load() - 1,
                                            type_: VertexType::LAST_HIT_LIGHT.into()
                                        }));
                                    },
                                    {
                                        if_!(
                                            !reconnection_vertex.valid()
                                                & !shift_mapping.is_base_path,
                                            {
                                                // the base path does not have a valid reconnection vertex
                                                // if a connectable vertex is found for shift path, it must be rejected
                                                if_!(can_connect, {
                                                    debug_sm_failed_line.store(line!());
                                                    break_();
                                                });
                                            }
                                        );
                                    }
                                );
                            }
                        );
                    }

                    if_!(depth.load().cmpge(self.max_depth), {
                        break_();
                    });
                    let u_bsdf = sampler.next_3d();
                    let sample = eval.bsdf.sample(surface, si, wo, u_bsdf);

                    // Direct Lighting
                    let (direct, direct_wi, direct_light_pdf) = if self.use_nee {
                        let (direct, direct_wi, direct_light_pdf) = if_!(!self.indirect_only | depth.load().cmpgt(1), {
                            let pn = PointNormalExpr::new(p, ng);
                            let sample = eval.light.sample(pn,sampler.next_3d());
                            let wi = sample.wi;
                            let (bsdf_f, bsdf_pdf) = eval.bsdf.evaluate_color_and_pdf(surface, si, wo, wi);
                            lc_assert!(bsdf_pdf.cmpge(0.0));
                            lc_assert!(bsdf_f.min().cmpge(0.0));
                            let w = mis_weight(sample.pdf, bsdf_pdf, 1);
                            let shadow_ray = sample.shadow_ray.set_exclude0(make_uint2(si.inst_id(), si.prim_id()));
                            let occluded = scene.occlude(shadow_ray);
                            // cpu_dbg!(sample.pdf);
                            let direct = sample.li / sample.pdf;
                            if_!(!occluded & sample.pdf.cmpgt(0.0), {
                                acc_radiance(bsdf_f * direct * w);
                                (direct, wi, sample.pdf)
                            }, else {
                                (Color::zero(eval.color_repr), wi, const_(0.0f32))
                            })
                        }, else {
                            (Color::zero(eval.color_repr), Float3Expr::zero(), const_(0.0f32))
                        });
                        (Some(direct), direct_wi, direct_light_pdf)
                    } else {
                        (None, Float3Expr::zero(), const_(0.0f32))
                    };
                    if let Some(shift_mapping) = &shift_mapping {
                        if_!(
                            !shift_mapping.is_base_path & shift_mapping.vertex.valid(),
                            {
                                if_!(depth.load().cmpgt(MIN_RECONNECT_DEPTH), {
                                    let dist = (prev_p.load() - p).length();
                                    let can_connect = dist.cmpgt(shift_mapping.min_dist)
                                        & sample.lobe_roughness.cmpgt(shift_mapping.min_roughness)
                                        & prev_roughness.load().cmpgt(shift_mapping.min_roughness);
                                    // cpu_dbg!(can_connect);
                                    if_!(can_connect, {
                                        debug_sm_failed_line.store(line!());
                                        break_();
                                    })
                                });
                                if_!(shift_mapping.vertex.depth().load().cmpeq(depth.load()), {
                                    let reconnection_vertex = shift_mapping.vertex.load();
                                    let vertex_type = reconnection_vertex.type_();
                                    let reconnect_si = scene.si_from_hitinfo(
                                        reconnection_vertex.inst_id(),
                                        reconnection_vertex.prim_id(),
                                        reconnection_vertex.bary(),
                                    );
                                    let reconnect_instance = scene
                                        .meshes
                                        .mesh_instances
                                        .var()
                                        .read(reconnection_vertex.inst_id());
                                    let reconnect_surface = reconnect_instance.surface();
                                    let dist = (p - reconnect_si.geometry().p()).length();
                                    let wi = (reconnect_si.geometry().p() - p).normalize();
                                    // cpu_dbg!(dist);
                                    let can_connect = dist.cmpgt(shift_mapping.min_dist)
                                        & sample.lobe_roughness.cmpgt(shift_mapping.min_roughness);
                                    // cpu_dbg!(can_connect);
                                    if_!(!can_connect, {
                                        debug_sm_failed_line.store(line!());
                                        break_();
                                    });

                                    let vis_ray = RayExpr::new(
                                        offset_ray_origin(p, face_forward(ng, wi)),
                                        wi,
                                        0.0,
                                        dist * (1.0 - 1e-3),
                                        make_uint2(si.inst_id(), si.prim_id()),
                                        make_uint2(reconnect_si.inst_id(), reconnect_si.prim_id()),
                                    );
                                    let occluded = scene.occlude(vis_ray);
                                    // cpu_dbg!(occluded);
                                    if_!(occluded, {
                                        debug_sm_failed_line.store(line!());
                                        break_();
                                    });
                                    // let cos_theta_y1 = ng.dot(wi).abs(); // should be included in bsdf
                                    let cos_theta_y2 = reconnect_si.geometry().ng().dot(wi).abs();
                                    let cos_theta_x2 = reconnect_si
                                        .geometry()
                                        .ng()
                                        .dot(reconnection_vertex.wo().unpack())
                                        .abs();
                                    if_!(cos_theta_y2.cmpeq(0.0), {
                                        debug_sm_failed_line.store(line!());
                                        break_();
                                    });
                                    lc_assert!(cos_theta_x2.cmpgt(0.0));
                                    // let geometry_term = cos_theta_y2 / dist.sqr();

                                    /*
                                     *
                                     * x_i                 x_{i+1}                               x_{i+2}
                                     *    vertex.prev_pdf          pdf(x_i -> x_{i+1}, x_{i+1} -> x_{i+2}) = pdf_x2
                                     * y_i                 y_{i+1}                               y_{i+2} =x_{i+2}
                                     *    pdf_y1                 pdf(x_i -> x_{i+1}, x_{i+1} -> x_{i+2}) = pdf_y2
                                     */
                                    let (f1, pdf_y1) =
                                        eval.bsdf.evaluate_color_and_pdf(surface, si, wo, wi);
                                    // cpu_dbg!(pdf_y1);
                                    if_!(pdf_y1.cmple(0.0), {
                                        debug_sm_failed_line.store(line!());
                                        break_();
                                    });
                                    let (f2, pdf_y2) = if_!(vertex_type.cmpeq(VertexType::LAST_HIT_LIGHT), {
                                        (Color::zero(eval.color_repr), const_(1.0f32))
                                    }, else {
                                        eval.bsdf.evaluate_color_and_pdf(reconnect_surface, reconnect_si, -wi, reconnection_vertex.wi().unpack())
                                    });
                                    // cpu_dbg!(f2.as_rgb());
                                    // cpu_dbg!(reconnection_vertex.wo());
                                    // cpu_dbg!(-wi);
                                    // scene.surfaces.get(reconnect_surface).dispatch(|_, k, _|{
                                    //     match k {
                                    //         PolyKey::Simple(s)=>{
                                    //             if s == "diffuse" {
                                    //                 cpu_dbg!(const_(1u32));
                                    //             }else {
                                    //                 cpu_dbg!(const_(2u32));
                                    //             }
                                    //         }
                                    //         _=>unreachable!()
                                    //     }
                                    // });
                                    // cpu_dbg!(make_float2(
                                    //     reconnect_si.frame().to_local(-wi).y(),
                                    //     reconnect_si.frame().to_local(reconnection_vertex.wi()).y()
                                    // ));
                                    // cpu_dbg!(make_float2(
                                    //     reconnect_si.frame().to_local(reconnection_vertex.wo()).y(),
                                    //     reconnect_si.frame().to_local(reconnection_vertex.wi()).y()
                                    // ));
                                    // cpu_dbg!(eval.bsdf.evaluate(reconnect_surface, reconnect_si, -wi, reconnection_vertex.wi()).as_rgb());
                                    if_!(pdf_y2.cmple(0.0), {
                                        debug_sm_failed_line.store(line!());
                                        break_();
                                    });
                                    let throughput = {
                                        let le = eval.light.le(vis_ray, reconnect_si);
                                        let light_pdf = eval
                                            .light
                                            .pdf(reconnect_si, PointNormalExpr::new(p, ng));
                                        let w = if self.use_nee {
                                            mis_weight(pdf_y1, light_pdf, 1)
                                        } else {
                                            const_(1.0f32)
                                        };
                                        let vertex_le = le * w;
                                        // cpu_dbg!(reconnection_vertex.direct_wi());
                                        let direct_f = if_!(reconnection_vertex.direct_wi().unpack().cmpne(0.0).any(), {
                                            let direct_wi = reconnection_vertex.direct_wi().unpack();
                                            let (f, bsdf_pdf) = eval.bsdf.evaluate_color_and_pdf(reconnect_surface, reconnect_si, -wi, direct_wi);
                                            let w = mis_weight(reconnection_vertex.direct_light_pdf(), bsdf_pdf, 1);
                                            f * w
                                        }, else {
                                            Color::zero(eval.color_repr)
                                        });
                                        // cpu_dbg!(direct_f.as_rgb());
                                        // cpu_dbg!(reconnection_vertex.indirect());
                                        // cpu_dbg!(reconnection_vertex.direct());
                                        f1 / pdf_y1
                                            * (vertex_le
                                                + direct_f
                                                    * Color::Rgb(
                                                        reconnection_vertex.direct().unpack(),
                                                    )
                                                + f2 * Color::Rgb(
                                                    reconnection_vertex.indirect().unpack(),
                                                ) / pdf_y2) // is this correct???
                                    };
                                    let pdf_x = if_!(vertex_type.cmpeq(VertexType::LAST_HIT_LIGHT), {
                                        reconnection_vertex.prev_bsdf_pdf()
                                    }, else {
                                        reconnection_vertex.prev_bsdf_pdf() * reconnection_vertex.bsdf_pdf()
                                    });
                                    let pdf_y = pdf_y1 * pdf_y2;
                                    // cpu_dbg!(pdf_y1);

                                    // cpu_dbg!(throughput.as_rgb());
                                    acc_radiance(throughput);

                                    let jacobian = pdf_y / pdf_x
                                        * (cos_theta_y2 / cos_theta_x2).abs()
                                        * (reconnection_vertex.dist() / dist).sqr();
                                    // lc_assert!(pdf_y.cmpgt(0.0));
                                    lc_assert!(pdf_x.cmpgt(0.0));
                                    // cpu_dbg!(jacobian);
                                    let jacobian =
                                        select(jacobian.is_finite(), jacobian, const_(0.0f32));
                                    shift_mapping.success.store(jacobian.cmpgt(0.0));
                                    shift_mapping.jacobian.store(jacobian);
                                    // if_!(shift_mapping.success.load(), {
                                    //     cpu_dbg!(jacobian)
                                    // });
                                    break_();
                                });
                            }
                        );
                    }
                    {
                        let f = &sample.color;
                        lc_assert!(f.min().cmpge(0.0));
                        if_!(sample.pdf.cmple(0.0) | !sample.valid, {
                            break_();
                        });
                        acc_beta(f / sample.pdf);
                    }
                    if_!(depth.load().cmpgt(MIN_RECONNECT_DEPTH), {
                        if let Some(shift_mapping) = &shift_mapping {
                            let reconnection_vertex = reconnection_vertex.unwrap();
                            let dist = (prev_p.load() - p).length();
                            let can_connect = dist.cmpgt(shift_mapping.min_dist)
                                & sample.lobe_roughness.cmpgt(shift_mapping.min_roughness)
                                & prev_roughness.load().cmpgt(shift_mapping.min_roughness);
                            found_reconnectible_vertex
                                .store(found_reconnectible_vertex.load() | can_connect);

                            if_!(
                                !reconnection_vertex.valid()
                                    & shift_mapping.is_base_path
                                    & can_connect,
                                {
                                    reconnection_vertex.store(struct_!(ReconnectionVertex {
                                        direct: direct
                                            .unwrap_or(Color::zero(eval.color_repr))
                                            .as_rgb()
                                            .pack(),
                                        direct_wi: direct_wi.pack(),
                                        direct_light_pdf: direct_light_pdf,
                                        indirect: Float3Expr::zero().pack(),
                                        inst_id: si.inst_id(),
                                        prim_id: si.prim_id(),
                                        bary: si.bary(),
                                        wo: wo.pack(),
                                        wi: sample.wi.pack(),
                                        prev_bsdf_pdf: prev_bsdf_pdf.load(),
                                        bsdf_pdf: sample.pdf,
                                        dist: dist,
                                        depth: depth.load() - 1,
                                        type_: VertexType::LAST_NEE.into()
                                    }));
                                    reconnect_beta.store(Color::one(eval.color_repr));
                                    reconnect_l.store(Color::zero(eval.color_repr));
                                }
                            );
                            if_!(
                                !reconnection_vertex.valid()
                                    & !shift_mapping.is_base_path
                                    & can_connect,
                                {
                                    // the base path does not have a valid reconnection vertex
                                    // if a connectable vertex is found for shift path, it must be rejected
                                    break_();
                                }
                            );
                        }
                    });

                    {
                        prev_bsdf_pdf.store(sample.pdf);
                        prev_n.store(ng);
                        prev_p.store(p);
                        prev_roughness.store(sample.lobe_roughness);
                        let ro = offset_ray_origin(p, face_forward(ng, sample.wi));
                        ray.store(RayExpr::new(
                            ro,
                            sample.wi,
                            0.0,
                            1e20,
                            make_uint2(si.inst_id(), si.prim_id()),
                            make_uint2(u32::MAX, u32::MAX),
                        ));
                    }
                    if_!(depth.load().cmpgt(self.rr_depth), {
                        let cont_prob = beta.load().max().clamp(0.0, 1.0) * 0.95;
                        if_!(sampler.next_1d().cmpgt(cont_prob), {
                            break_();
                        }, else {
                            acc_beta(Color::one(eval.color_repr) / cont_prob);
                        });
                    });
                },
                {
                    if_!(!self.indirect_only | depth.load().cmpgt(1), {
                        acc_radiance(scene.env_map(ray.d().load(), eval));
                    });
                    break_();
                }
            );
        });

        if let Some(shift_mapping) = &shift_mapping {
            let reconnection_vertex = reconnection_vertex.unwrap();
            // if_!(
            //     !shift_mapping.is_base_path & debug_sm_failed_line.load().cmpne(0),
            //     {
            //         cpu_dbg!(debug_sm_failed_line.load());
            //     }
            // );
            if_!(
                reconnection_vertex.valid()
                    & reconnection_vertex
                        .type_()
                        .load()
                        .cmpne(VertexType::LAST_HIT_LIGHT)
                    & shift_mapping.is_base_path,
                {
                    reconnection_vertex.set_indirect(reconnect_l.load().as_rgb());
                }
            );
            if_!(
                !reconnection_vertex.valid() & !shift_mapping.is_base_path,
                {
                    shift_mapping
                        .success
                        .store(!found_reconnectible_vertex.load());
                    shift_mapping.jacobian.store(select(
                        found_reconnectible_vertex.load(),
                        const_(0.0f32),
                        const_(1.0f32),
                    ));
                }
            );
        }
        l.load().remove_nan().clamp(1e6)
    }
}
impl Integrator for PathTracer {
    fn render(&self, scene: Arc<Scene>, film: &mut Film, _options: &RenderOptions) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig:{:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        let npixels = resolution.x as usize * resolution.y as usize;
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let rngs = init_pcg32_buffer_with_seed(self.device.clone(), npixels, self.seed);
        let color_repr = ColorRepr::Rgb;
        let evaluators = scene.evaluators(color_repr);
        let kernel = self.device.create_kernel::<(u32, Int2)>(
            &|spp_per_pass: Expr<u32>, pixel_offset: Expr<Int2>| {
                let p = dispatch_id().xy();
                let i = p.x() + p.y() * resolution.x;
                let rngs = rngs.var();
                let sampler = IndependentSampler {
                    state: var!(Pcg32, rngs.read(i)),
                };
                for_range(const_(0)..spp_per_pass.int(), |_| {
                    let ip = p.int();
                    let shifted = ip + pixel_offset;
                    let shifted = shifted.clamp(0, const_(resolution).int() - 1).uint();
                    let (ray, ray_color, ray_w) =
                        scene
                            .camera
                            .generate_ray(film.filter(), shifted, &sampler, color_repr);
                    let l = self.radiance(&scene, ray, &sampler, &evaluators) * ray_color;
                    film.add_sample(p.float(), &l, ray_w);
                });
                rngs.write(i, sampler.state.load());
            },
        );
        let stream = self.device.default_stream();
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;
        stream.with_scope(|s| {
            while cnt < self.spp {
                let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
                let mut cmds = vec![];
                let tic = Instant::now();
                cmds.push(kernel.dispatch_async(
                    [resolution.x, resolution.y, 1],
                    &cur_pass,
                    &self.pixel_offset,
                ));
                s.submit(cmds);
                s.synchronize();
                let toc = Instant::now();
                acc_time += toc.duration_since(tic).as_secs_f64();
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
            }
        });
        progress.finish();
        log::info!("Rendering finished in {:.2}s", acc_time);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    film: &mut Film,
    config: &Config,
    options: &RenderOptions,
) {
    let pt = PathTracer::new(device.clone(), config.clone());
    pt.render(scene, film, options);
}
