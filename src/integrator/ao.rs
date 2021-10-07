// use crate::bsdf::*;
// use crate::camera::*;
use crate::film::*;
use crate::integrator::*;
use crate::sampler::*;
use crate::scene::*;
use crate::shape::*;
use crate::*;
pub struct RTAO {
    pub spp: u32,
}
impl Integrator for RTAO {
    fn render(&mut self, scene: &Scene) -> Film {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = RwLock::new(Film::new(&scene.camera.resolution()));
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            for _ in 0..self.spp {
                let (mut ray, _ray_weight) = scene.camera.generate_ray(&pixel, &mut sampler);
                let mut li = Spectrum::zero();
                {
                    if let Some(isct) = scene.intersect(&ray) {
                        // li = Spectrum { samples: isct.ng }
                        let ng = isct.ng;
                        let frame = Frame::from_normal(&ng);
                        let wi = {
                            let w = consine_hemisphere_sampling(&sampler.next2d());
                            frame.to_world(&w)
                        };
                        // li = Spectrum {
                        //     samples: vec3(1.0,1.0,1.0) * glm::dot(&wi, &vec3(0.2, 0.8, 0.0)),
                        // };

                        // li = Spectrum{samples:wi};
                        let p = ray.at(isct.t);
                        ray = Ray::spawn(&p, &wi).offset_along_normal(&ng);
                        if !scene.occlude(&ray) {
                            li = Spectrum::one();
                        }
                    }
                }
                acc_li = acc_li + li;
            }
            acc_li = acc_li / (self.spp as Float);
            {
                let film = &mut film.write().unwrap();
                film.add_sample(&uvec2(x, y), &acc_li, 1.0);
            }
        });
        film.into_inner().unwrap()
    }
}
