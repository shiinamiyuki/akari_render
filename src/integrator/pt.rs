use luisa::rtx::offset_ray_origin;
use rand::{thread_rng, Rng};

use super::Integrator;

use crate::{
    color::*, film::*, geometry::*, interaction::*, sampler::*, scene::*, surface::Bsdf, *,
};
pub struct PathTracer {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
}

impl PathTracer {
    pub fn new(device: Device, spp: u32, spp_per_pass: u32, max_depth: u32, use_nee: bool) -> Self {
        Self {
            device,
            spp,
            max_depth,
            spp_per_pass,
            use_nee,
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
        scene: &Scene,
        ray: Expr<Ray>,
        sampler: &dyn Sampler,
        color_repr: &ColorRepr,
    ) -> Color {
        let l = ColorVar::zero(color_repr);
        let beta = ColorVar::one(color_repr);
        let ray = var!(Ray, ray);
        let depth = var!(u32, 0);
        let ctx = ShadingContext {
            scene,
            color_repr: color_repr.clone(),
        };
        let prev_bsdf_pdf = var!(f32);
        let prev_n = var!(Float3, ray.d());
        loop_!({
            let si = scene.intersect(ray.load());
            let wo = -ray.d().load();
            if_!(si.valid(), {

                let inst_id = si.inst_id();
                let instance = scene.meshes.mesh_instances.var().read(inst_id);
                if_!(instance.light().valid(), {
                    let direct = scene.lights.le(ray.load(), si, &ctx);
                    if_!(depth.load().cmpeq(0) | !self.use_nee, {
                        l.store(&(l.load() + beta.load() * &direct));
                    }, else {
                        let pn = {
                            let p = ray.o();
                            let n = prev_n.load();
                            PointNormalExpr::new(p, n)
                        };
                        let (light_pdf, _) = scene.lights.pdf_direct(si, pn, &ctx);
                        let w = mis_weight(prev_bsdf_pdf.load(), light_pdf, 1);
                        l.store(&(l.load() + beta.load() * &direct * w));
                    });
                });
                let p = si.geometry().p();
                let ng = si.geometry().ng();
                let surface = instance.surface();
                let surface = scene.surfaces.get(surface);
                depth.store(depth.load() + 1);
                if_!(depth.load().cmpge(self.max_depth), {
                    break_();
                });
                // Direct Lighting
                if self.use_nee {
                    let pn = PointNormalExpr::new(p, ng);
                    let (sample, _) = scene.lights.sample_direct(pn, sampler.next_2d(), &ctx);
                    let wi = sample.wi;
                    let (bsdf_f, bsdf_pdf) = surface.dispatch(|_tag, _key, surface| {
                        let bsdf = surface.closure(si, &ctx);
                        let pdf = bsdf.pdf(wo, wi, &ctx);
                        let f = bsdf.evaluate(wo, wi, &ctx);
                        (f, pdf)
                    });
                    let w = mis_weight(sample.pdf, bsdf_pdf, 1);
                    let occluded = scene.occlude(sample.shadow_ray);
                    // cpu_dbg!(sample.pdf);
                    if_!(!occluded, {
                        l.store(&(l.load() + beta.load() * bsdf_f * &sample.li / sample.pdf * w));
                    });
                }

                // BSDF sampling
                surface.dispatch(|_tag, _key, surface| {
                    let bsdf = surface.closure(si, &ctx);
                    let sample = bsdf.sample(wo, sampler.next_1d(), sampler.next_2d(), &ctx);
                    let wi = sample.wi;
                    let f = &sample.color;
                    beta.store(&(beta.load() * f / sample.pdf));
                    prev_bsdf_pdf.store(sample.pdf);
                    let ro = offset_ray_origin(p, ng);
                    ray.store(RayExpr::new(ro, wi, 1e-3, 1e20));
                });
            }, else {
                break_();

            });
        });
        l.load()
    }
}
impl Integrator for PathTracer {
    fn render(&self, scene: &Scene, film: &mut Film) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}, spp: {}",
            resolution.x,
            resolution.y,
            self.spp
        );
        let npixels = resolution.x as usize * resolution.y as usize;
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        film.clear();
        let mut rng = thread_rng();
        let seeds = self
            .device
            .create_buffer_from_fn(npixels, |_| rng.gen::<u32>());
        let kernel = self
            .device
            .create_kernel::<(u32,)>(&|spp_per_pass: Expr<u32>| {
                let p = dispatch_id().xy();
                let i = p.x() + p.y() * resolution.x;
                let seeds = seeds.var();
                let seed = seeds.read(i);
                let sampler = LcgSampler {
                    state: var!(u32, seed),
                };
                for_range(const_(0)..spp_per_pass.int(), |_| {
                    let color_repr = ColorRepr::Rgb;
                    let (ray, ray_color, ray_w) =
                        scene.camera.generate_ray(p, &sampler, &color_repr);
                    let l = self.radiance(scene, ray, &sampler, &color_repr) * ray_color;
                    film.add_sample(p.float(), &l, ray_w);
                });
                seeds.write(i, sampler.state.load());
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
