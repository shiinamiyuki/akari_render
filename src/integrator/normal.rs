use rand::{thread_rng, Rng};

use super::Integrator;

use crate::{color::*, film::*, geometry::*, interaction::*, sampler::*, scene::*, *};
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
        let kernel = self.device.create_kernel::<(u32,)>(&|_spp: Expr<u32>| {
            let p = dispatch_id().xy();
            let i = p.x() + p.y() * resolution.x;
            let seeds = seeds.var();
            let seed = seeds.read(i);
            let sampler = LcgSampler {
                state: var!(u32, seed),
            };
            let color_repr = ColorRepr::Rgb;
            let (ray, ray_color, ray_w) = scene.camera.generate_ray(p, &sampler, &color_repr);
            let si = scene.intersect(ray);
            // cpu_dbg!(ray);
            let color = if_!(si.valid(), {
                let ns = si.geometry().ng();
                // cpu_dbg!(make_uint2(si.inst_id(), si.prim_id()));
                Color::Rgb(ns * 0.5 + 0.5) * ray_color
                // Color::Rgb(make_float3(si.bary().x(),si.bary().y(), 1.0))
            }, else {
                Color::zero(&color_repr)
            });
            film.add_sample(p.float(), &color, ray_w);
            seeds.write(i, sampler.state.load());
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

