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
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct PathVertex {
    pub wo: Float3,
    pub si: SurfaceInteraction,
}

impl PathTracer {
    pub fn new(device: Device, spp: u32, spp_per_pass: u32, max_depth: u32) -> Self {
        Self {
            device,
            spp,
            max_depth,
            spp_per_pass,
        }
    }
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
        while_!(Bool::from(true), {
            let si = scene.intersect(ray.load());
            let wo = -ray.d().load();
            if_!(si.valid(), {
                let inst_id = si.inst_id();
                let instance = scene.meshes.mesh_instances.var().read(inst_id);
                if_!(instance.light().valid(), {
                    let light = scene.lights.get(instance.light());
                    let direct = light.dispatch(|_tag, _key, light|light.le(ray.load(), si, &ctx));
                    l.store(&(l.load() + beta.load() * direct));
                });
                let p = si.geometry().p();
                let ns = si.geometry().ns();
                let ng = si.geometry().ng();
                let surface = instance.surface();
                let surface = scene.surfaces.get(surface);
                depth.store(depth.load() + 1);
                if_!(depth.load().cmpge(self.max_depth), {
                    break_();
                });
                surface.dispatch(|_tag, _key, surface| {
                    let bsdf = surface.closure(si, &ctx);
                    let sample = bsdf.sample(wo, sampler.next_1d(), sampler.next_2d(), &ctx);
                    let wi = sample.wi;
                    let f = &sample.color;
                    beta.store(&(beta.load() * f * wi.dot(ns).abs() / sample.pdf));
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
    fn render(&self, scene: &Scene, film: &mut Film) -> luisa::Result<()> {
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
            .create_buffer_from_fn(npixels, |_| rng.gen::<u32>())?;
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
            })?;
        let stream = self.device.default_stream();
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        stream.with_scope(|s| -> luisa::Result<()> {
          
            while cnt < self.spp {
                let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
                let mut cmds = vec![];
                cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1], &cur_pass));
                s.submit(cmds)?;
                s.synchronize()?;
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
            }
            
            Ok(())
        })?;
        progress.finish();
        Ok(())
    }
}
