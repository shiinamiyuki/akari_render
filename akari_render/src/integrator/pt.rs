use std::{sync::Arc, time::Instant};

use luisa::rtx::offset_ray_origin;
use rand::Rng;

use super::{Integrator, RenderOptions};
use crate::{
    color::*, film::*, geometry::*, interaction::SurfaceInteraction, sampler::*, scene::*,
    svm::surface::*, *,
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
        ray: Expr<Ray>,
        sampler: &dyn Sampler,
        eval: &Evaluators,
    ) -> Color {
        let si = scene.intersect(ray);
        if_!(
            si.valid(),
            {
                let n = si.geometry().ng();
                // cpu_dbg!(n);
                let c = n;// * 0.5 + 0.5;
                // cpu_dbg!(c);
                Color::from_flat(eval.color_repr(), make_float4(c.x(), c.y(), c.z(), 0.0))
            },
            { Color::zero(eval.color_repr()) }
        )
    }
}
impl Integrator for PathTracer {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        _options: &RenderOptions,
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
        let evaluators = scene.evaluators(color_pipeline);
        let kernel = self.device.create_kernel::<(u32, Int2)>(
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
                    let l = self.radiance(&scene, ray, sampler, &evaluators) * ray_color;
                    film.add_sample(p.float(), &l, ray_w);
                });
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
    sampler: SamplerConfig,
    color_pipeline: ColorPipeline,
    film: &mut Film,
    config: &Config,
    options: &RenderOptions,
) {
    let pt = PathTracer::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
