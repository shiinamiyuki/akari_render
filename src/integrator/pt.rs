use std::sync::Arc;

use luisa::rtx::offset_ray_origin;
use rand::Rng;

use super::Integrator;
use crate::{color::*, film::*, geometry::*, sampler::*, scene::*, surface::Bsdf, *};
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
    config: Config,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
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
impl PathTracer {
    pub fn radiance(
        &self,
        scene: &Arc<Scene>,
        ray: Expr<Ray>,
        sampler: &dyn Sampler,
        eval: &Evaluators,
    ) -> Color {
        let color_repr = eval.color_repr;
        let l = ColorVar::zero(color_repr);
        let beta = ColorVar::one(color_repr);
        let ray = var!(Ray, ray);
        let depth = var!(u32, 0);
        let prev_bsdf_pdf = var!(f32);
        let prev_n = var!(Float3, ray.d());
        loop_!({
            let si = scene.intersect(ray.load());
            let wo = -ray.d().load();
            if_!(si.valid(), {

                let inst_id = si.inst_id();
                let instance = scene.meshes.mesh_instances.var().read(inst_id);

                if_!(instance.light().valid() & (!self.indirect_only | depth.load().cmpgt(1)), {
                    let direct = eval.light.le(ray.load(), si);

                    if_!(depth.load().cmpeq(0) | !self.use_nee, {
                        l.store(l.load() + beta.load() * &direct);
                    }, else {
                        let pn = {
                            let p = ray.o();
                            let n = prev_n.load();
                            PointNormalExpr::new(p, n)
                        };
                        let light_pdf = eval.light.pdf(si, pn);
                        let w = mis_weight(prev_bsdf_pdf.load(), light_pdf, 1);
                        l.store(l.load() + beta.load() * &direct * w);
                    });
                });
                let p = si.geometry().p();
                let ng = si.geometry().ng();
                let surface = instance.surface();
                depth.store(depth.load() + 1);
                if_!(depth.load().cmpge(self.max_depth), {
                    break_();
                });
                // Direct Lighting
                if self.use_nee {
                    if_!(!self.indirect_only | depth.load().cmpgt(1), {
                        let pn = PointNormalExpr::new(p, ng);
                        let sample = eval.light.sample(pn,sampler.next_3d());
                        let wi = sample.wi;
                        let (bsdf_f, bsdf_pdf) = eval.bsdf.evaluate_color_and_pdf(surface, si, wo, wi);
                        let w = mis_weight(sample.pdf, bsdf_pdf, 1);
                        let occluded = scene.occlude(sample.shadow_ray);
                        // cpu_dbg!(sample.pdf);
                        if_!(!occluded & sample.pdf.cmpgt(0.0), {
                            l.store(l.load() + beta.load() * bsdf_f * &sample.li / sample.pdf * w);
                        });
                    });

                }
                // BSDF sampling
                {
                    let sample = eval.bsdf.sample(surface, si, wo,sampler.next_3d());
                    let wi = sample.wi;
                    let f = &sample.color;
                    if_!(sample.pdf.cmple(0.0) | !sample.valid,{
                        break_();
                    });
                    beta.store(beta.load() * f / sample.pdf);

                    prev_bsdf_pdf.store(sample.pdf);
                    let ro = offset_ray_origin(p, ng);
                    ray.store(RayExpr::new(ro, wi, 1e-3, 1e20));
                }
                if_!(depth.load().cmpgt(self.rr_depth), {
                    let cont_prob = beta.load().max().clamp(0.0, 1.0) * 0.95;
                    if_!(sampler.next_1d().cmpgt(cont_prob), {
                        break_();
                    }, else {
                        beta.store(beta.load() / cont_prob);
                    });
                });
            }, else {
                break_();

            });
        });
        l.load().remove_nan().clamp(1e6)
    }
}
impl Integrator for PathTracer {
    fn render(&self, scene: Arc<Scene>, film: &mut Film) {
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
        let rngs = init_pcg32_buffer(self.device.clone(), npixels);
        let color_repr = ColorRepr::Rgb;
        let evaluators = scene.evaluators(color_repr);
        let kernel = self
            .device
            .create_kernel::<(u32,)>(&|spp_per_pass: Expr<u32>| {
                let p = dispatch_id().xy();
                let i = p.x() + p.y() * resolution.x;
                let rngs = rngs.var();
                let sampler = IndependentSampler {
                    state: var!(Pcg32, rngs.read(i)),
                };
                for_range(const_(0)..spp_per_pass.int(), |_| {
                    let (ray, ray_color, ray_w) =
                        scene.camera.generate_ray(p, &sampler, color_repr);
                    let l = self.radiance(&scene, ray, &sampler, &evaluators) * ray_color;
                    film.add_sample(p.float(), &l, ray_w);
                });
                rngs.write(i, sampler.state.load());
            });
        let stream = self.device.default_stream();
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        stream.with_scope(|s| {
            while cnt < self.spp {
                let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
                let mut cmds = vec![];
                cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1], &cur_pass));
                s.submit(cmds);
                s.synchronize();
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
            }
        });
        progress.finish();
    }
}

pub fn render(device: Device, scene: Arc<Scene>, film: &mut Film, config: &Config) {
    let pt = PathTracer::new(device.clone(), config.clone());
    pt.render(scene, film);
}
