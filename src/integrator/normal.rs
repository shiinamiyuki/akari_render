use std::sync::Arc;

use super::{Integrator, RenderOptions};

use crate::{color::*, film::*, sampler::*, scene::*, *};
pub struct NormalVis {
    device: Device,
    pub spp: u32,
}
impl NormalVis {
    pub fn new(device: Device, spp: u32) -> Self {
        Self { device, spp }
    }
}

impl Integrator for NormalVis {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler: SamplerConfig,
        color_repr: ColorRepr,
        film: &mut Film,
        _options: &RenderOptions,
    ) {
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
        let sampler_creator =
            IndependentSamplerCreator::new(self.device.clone(), scene.camera.resolution(), 0);
        let kernel = self.device.create_kernel::<(u32,)>(&|_spp: Expr<u32>| {
            let p = dispatch_id().xy();
            let sampler = sampler_creator.create(p);
            let sampler = sampler.as_ref();
            sampler.start();
            let (ray, ray_color, ray_w) =
                scene
                    .camera
                    .generate_ray(film.filter(), p, sampler, color_repr);
            let si = scene.intersect(ray);
            // cpu_dbg!(ray);
            let color = if_!(si.valid(), {
                let ns = si.geometry().ng();
                // cpu_dbg!(make_uint2(si.inst_id(), si.prim_id()));
                Color::Rgb(ns * 0.5 + 0.5, color_repr.rgb_colorspace().unwrap()) * ray_color
                // Color::Rgb(make_float3(si.bary().x(),si.bary().y(), 1.0))
            }, else {
                Color::zero(color_repr)
            });
            film.add_sample(p.float(), &color, ray_w);
        });
        let stream = self.device.default_stream();
        stream.with_scope(|s| {
            let mut cmds = vec![];
            for _ in 0..self.spp {
                cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1], &self.spp));
            }
            s.submit(cmds);
            s.synchronize();
        });
    }
}
