use util::parallel_for;

// use crate::bsdf::*;
// use crate::camera::*;
use crate::film::*;
use crate::sampler::*;
use crate::scene::*;
use crate::*;
pub struct NormalVis {}
impl Integrator for NormalVis {
    fn render(&self, scene: &Scene) -> Film {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler {
                rng: Pcg::new(id as u64),
            };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);

            let (ray, _ray_weight) = scene.camera.generate_ray(pixel, &mut sampler);

            if let Some(si) = scene.intersect(&ray) {
                let ns = si.ns;
                film.add_sample(uvec2(x, y), SampledSpectrum::from_rgb_linear(ns * 0.5 + 0.5), 1.0);
            }
        });
        film
    }
}
