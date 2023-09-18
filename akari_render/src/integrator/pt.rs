use std::{sync::Arc, time::Instant};

use luisa::rtx::offset_ray_origin;
use rand::Rng;

use super::{Integrator, RenderSession};
use crate::{
    color::*, film::*, geometry::*, interaction::SurfaceInteraction, mesh::MeshInstance,
    sampler::*, scene::*, svm::surface::*, *,
};
use serde::{Deserialize, Serialize};


pub struct PathTracerBase<'a> {
    pub max_depth: Expr<u32>,
    pub use_nee: bool,
    pub rr_depth: Expr<u32>,
    pub indirect_only: bool,
    pub radiance: ColorVar,
    pub beta: ColorVar,
    pub color_pipeline: ColorPipeline,
    pub depth: Var<u32>,
    pub si: Var<SurfaceInteraction>,
    pub prev_bsdf_pdf: Var<f32>,
    pub prev_ng: Var<Float3>,
    pub swl: Var<SampledWavelengths>,
    pub eval: &'a Evaluators,
    pub scene: &'a Scene,

    //
    pub instance: Var<MeshInstance>,
    pub p: Var<Float3>,
    pub wo: Var<Float3>,
    pub ns: Var<Float3>,
    pub ng: Var<Float3>,
}
#[derive(Aggregate)]
pub struct DirectLighting {
    pub irradiance: Color,
    pub bsdf_f: Color,
    pub weight: Expr<f32>,
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub shadow_ray: Expr<Ray>,
    pub valid: Expr<bool>,
}
impl DirectLighting {
    pub fn invalid(color_pipeline: ColorPipeline) -> Self {
        Self {
            irradiance: Color::zero(color_pipeline.color_repr),
            wi: Float3Expr::zero(),
            pdf: const_(0.0f32),
            weight: const_(0.0f32),
            shadow_ray: zeroed::<Ray>(),
            valid: const_(false),
            bsdf_f: Color::zero(color_pipeline.color_repr),
        }
    }
}

impl<'a> PathTracerBase<'a> {
    pub fn new(
        scene: &'a Scene,
        eval: &'a Evaluators,
        max_depth: Expr<u32>,
        rr_depth: Expr<u32>,
        use_nee: bool,
        indirect_only: bool,
        swl: Var<SampledWavelengths>,
    ) -> Self {
        let color_pipeline = eval.color_pipeline;
        Self {
            max_depth,
            use_nee,
            rr_depth,
            indirect_only,
            radiance: ColorVar::zero(color_pipeline.color_repr),
            beta: ColorVar::one(color_pipeline.color_repr),
            color_pipeline,
            eval,
            scene,
            depth: var!(u32, 0),
            si: var!(SurfaceInteraction),
            prev_bsdf_pdf: var!(f32),
            prev_ng: var!(Float3),
            swl,
            instance: var!(MeshInstance),
            p: var!(Float3),
            wo: var!(Float3),
            ns: var!(Float3),
            ng: var!(Float3),
        }
    }
    pub fn add_radiance(&self, r: Color) {
        self.radiance
            .store(self.radiance.load() + self.beta.load() * r);
    }
    pub fn mul_beta(&self, r: Color) {
        self.beta.store(self.beta.load() * r);
    }
    pub fn set_si(&self, si: Expr<SurfaceInteraction>, wo: Expr<Float3>) {
        self.si.store(si);
        let g = si.geometry();
        *self.p.get_mut() = g.p();
        *self.ns.get_mut() = g.ns();
        *self.ng.get_mut() = g.ng();
        *self.wo.get_mut() = wo;
        let inst_id = si.inst_id();
        let instance = self.scene.meshes.mesh_instances.var().read(inst_id);
        *self.instance.get_mut() = instance;
    }
    pub fn next_intersection(&self, ray: Expr<Ray>) -> Expr<bool> {
        let si = self.scene.intersect(ray);
        if_!(
            si.valid(),
            {
                self.set_si(si, -ray.d());
                const_(true)
            },
            else,
            { const_(false) }
        )
    }
    pub fn sample_surface(&self, u_bsdf: Expr<Float3>) -> BsdfSample {
        let surface = *self.instance.surface();
        let sample = self
            .eval
            .surface
            .sample(surface, *self.si, *self.wo, u_bsdf, self.swl);
        sample
    }
    pub fn sample_light(&self, u: Expr<Float3>) -> DirectLighting {
        if self.use_nee {
            if_!(
                !self.indirect_only | self.depth.load().cmpgt(1),
                {
                    let p = *self.p;
                    let ng = *self.ng;
                    let pn = PointNormalExpr::new(p, ng);
                    let eval = self.eval;
                    let sample = eval.light.sample(pn, u, *self.swl);
                    let wi = sample.wi;
                    let surface = *self.instance.surface();
                    let wo = *self.wo;
                    let (bsdf_f, bsdf_pdf) = eval
                        .surface
                        .evaluate_color_and_pdf(surface, *self.si, wo, wi, *self.swl);
                    lc_assert!(bsdf_pdf.cmpge(0.0));
                    lc_assert!(bsdf_f.min().cmpge(0.0));
                    let w = mis_weight(sample.pdf, bsdf_pdf, 1);
                    let shadow_ray = sample
                        .shadow_ray
                        .set_exclude0(make_uint2(*self.si.inst_id(), *self.si.prim_id()));
                    DirectLighting {
                        weight: w,
                        irradiance: sample.li,
                        wi,
                        pdf: sample.pdf,
                        shadow_ray,
                        valid: const_(true),
                        bsdf_f,
                    }
                },
                else,
                { DirectLighting::invalid(self.color_pipeline) }
            )
        } else {
            DirectLighting::invalid(self.color_pipeline)
        }
    }
    pub fn continue_prob(&self) -> (Expr<bool>, Expr<f32>) {
        let depth = &self.depth;
        let beta = &self.beta;
        if_!(
            depth.load().cmpgt(self.rr_depth),
            {
                let cont_prob = beta.load().max().clamp(0.0, 1.0) * 0.95;
                (true.into(), cont_prob)
            },
            { (false.into(), const_(1.0f32)) }
        )
    }
    pub fn hit_envmap(&self, ray: Expr<Ray>) -> (Color, Expr<f32>) {
        (Color::zero(self.color_pipeline.color_repr), const_(0.0f32))
    }
    pub fn handle_surface_light(&self, ray: Expr<Ray>) -> (Color, Expr<f32>) {
        let instance = *self.instance;
        let eval = self.eval;
        let depth = &self.depth;

        if_!(
            instance.light().valid() & (!self.indirect_only | depth.cmpgt(1)),
            {
                let si = *self.si;
                let direct = eval.light.le(ray, si, *self.swl);
                // cpu_dbg!(direct.flatten());
                if_!(depth.cmpeq(0) | !self.use_nee, {
                   (direct, const_(1.0f32))
                }, else {
                    let pn = {
                        let p = ray.o();
                        let n = *self.prev_ng;
                        PointNormalExpr::new(p, n)
                    };
                    let light_pdf = eval.light.pdf(si, pn, *self.swl);
                    let w = mis_weight(*self.prev_bsdf_pdf, light_pdf, 1);

                    (direct, w)
                })
            },
            else,
            { (Color::zero(self.color_pipeline.color_repr), const_(0.0f32)) }
        )
    }

    pub fn run_megakernel(&self, ray: Expr<Ray>, sampler: &dyn Sampler) {
        let ray = var!(Ray, ray);
        loop_!({
            let hit = self.next_intersection(*ray);
            if_!(!hit, {
                let (direct, w) = self.hit_envmap(*ray);
                self.add_radiance(direct * w);
                break_();
            });
            {
                let (direct, w) = self.handle_surface_light(*ray);
                self.add_radiance(direct * w);
            }

            if_!(self.depth.load().cmpge(self.max_depth), {
                break_();
            });
            *self.depth.get_mut() += 1;

            let direct_lighting = self.sample_light(sampler.next_3d());
            if_!(direct_lighting.valid, {
                let shadow_ray = direct_lighting.shadow_ray;
                if_!(!self.scene.occlude(shadow_ray), {
                    let direct = direct_lighting.irradiance
                        * direct_lighting.bsdf_f
                        * direct_lighting.weight
                        / direct_lighting.pdf;
                    self.add_radiance(direct);
                });
            });
            let bsdf_sample = self.sample_surface(sampler.next_3d());
            let f = &bsdf_sample.color;
            lc_assert!(f.min().cmpge(0.0));
            if_!(bsdf_sample.pdf.cmple(0.0) | !bsdf_sample.valid, {
                break_();
            });

            self.mul_beta(f / bsdf_sample.pdf);
            let (rr_effective, cont_prob) = self.continue_prob();
            if_!(rr_effective, {
                let rr = sampler.next_1d().cmpge(cont_prob);
                if_!(rr, {
                    break_();
                });
                self.mul_beta(Color::one(self.color_pipeline.color_repr) / cont_prob);
            });
            {
                *self.prev_bsdf_pdf.get_mut() = bsdf_sample.pdf;
                *self.prev_ng.get_mut() = *self.ng;
                let ro = offset_ray_origin(*self.p, face_forward(*self.ng, bsdf_sample.wi));
                *ray.get_mut() = RayExpr::new(
                    ro,
                    bsdf_sample.wi,
                    0.0,
                    1e20,
                    make_uint2(*self.si.inst_id(), *self.si.prim_id()),
                    make_uint2(u32::MAX, u32::MAX),
                );
            }
        })
    }
}
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
        eval: &Evaluators,
        ray: Expr<Ray>,
        swl: Var<SampledWavelengths>,
        sampler: &dyn Sampler,
    ) -> Color {
        let pt = PathTracerBase::new(
            scene,
            eval,
            self.max_depth.into(),
            self.rr_depth.into(),
            self.use_nee,
            self.indirect_only,
            swl,
        );
        pt.run_megakernel(ray, sampler);
        pt.radiance.load()
    }
}
impl Integrator for PathTracer {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        session: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig:{:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let sampler_creator = sampler_config.creator(self.device.clone(), &scene, self.spp);
        let evaluators = scene.evaluators(color_pipeline, ADMode::None);
        let kernel = self.device.create_kernel::<fn(u32, Int2)>(
            &|spp_per_pass: Expr<u32>, pixel_offset: Expr<Int2>| {
                let p = dispatch_id().xy();
                let sampler = sampler_creator.create(p);
                let sampler = sampler.as_ref();
                for_range(const_(0)..spp_per_pass.int(), |_| {
                    sampler.start();
                    let ip = p.int();
                    let shifted = ip + pixel_offset;
                    let shifted = shifted.clamp(0, const_(resolution).int() - 1).uint();
                    let swl = sample_wavelengths(color_pipeline.color_repr, sampler);
                    let (ray, ray_color, ray_w) = scene.camera.generate_ray(
                        film.filter(),
                        shifted,
                        sampler,
                        color_pipeline.color_repr,
                        swl,
                    );
                    let swl = def(swl);
                    let l = self.radiance(&scene, &evaluators, ray, swl, sampler) * ray_color;
                    film.add_sample(p.float(), &l, *swl, ray_w);
                });
            },
        );
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;
        let update = || {
            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };

        while cnt < self.spp {
            let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
            let tic = Instant::now();
            kernel.dispatch(
                [resolution.x, resolution.y, 1],
                &cur_pass,
                &self.pixel_offset,
            );
            let toc = Instant::now();
            acc_time += toc.duration_since(tic).as_secs_f64();
            update();
            progress.inc(cur_pass as u64);
            cnt += cur_pass;
        }

        progress.finish();
        log::info!("Rendering finished in {:.2}s", acc_time);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    sampler: SamplerConfig,
    color_pipeline: ColorPipeline,
    film: &mut Film,
    config: &Config,
    options: &RenderSession,
) {
    let pt = PathTracer::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
