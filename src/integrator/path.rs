use crate::bsdf::*;
// use crate::camera::*;
use crate::film::*;
use crate::integrator::*;
use crate::light::*;
use crate::sampler::*;
use crate::scene::*;
use crate::shape::*;
use crate::texture::ShadingPoint;
use crate::*;
pub struct PathTracer {
    pub spp: u32,
    pub max_depth: u32,
}
fn mis_weight(mut pdf_a: Float, mut pdf_b: Float) -> Float {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    pdf_a / (pdf_a + pdf_b)
}
impl Integrator for PathTracer {
    fn render(&mut self, scene: &Scene) -> Film {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        parallel_for(npixels, 256, |id| {
            let mut sampler = SobolSampler::new(id as u32);
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            let mut prev_n: Option<Vec3> = None;
            let mut prev_bsdf_pdf: Option<Float> = None;
            for _ in 0..self.spp {
                let (mut ray, _ray_weight) = scene.camera.generate_ray(&pixel, &mut sampler);
                let mut li = Spectrum::zero();
                let mut beta = Spectrum::one();
                {
                    let mut depth = 0;
                    loop {
                        if let Some(isct) = scene.shape.intersect(&ray) {
                            let ng = isct.ng;
                            let frame = Frame::from_normal(&ng);
                            let shape = isct.shape.unwrap();
                            let opt_bsdf = shape.bsdf();
                            if opt_bsdf.is_none() {
                                break;
                            }
                            let p = ray.at(isct.t);
                            let bsdf = BsdfClosure {
                                sp: ShadingPoint::from_intersection(&isct),
                                frame,
                                bsdf: opt_bsdf.unwrap(),
                            };

                            if let Some(light) = scene.get_light_of_shape(shape) {
                                // li += beta * light.le(&ray);
                                if depth == 0 {
                                    li += beta * light.le(&ray);
                                } else {
                                    let light_pdf = scene.light_distr.pdf(light)
                                        * light
                                            .pdf_li(
                                                &ray.d,
                                                &ReferencePoint {
                                                    p: ray.o,
                                                    n: prev_n.unwrap(),
                                                },
                                            )
                                            .1;
                                    let bsdf_pdf = prev_bsdf_pdf.unwrap();
                                    assert!(light_pdf.is_finite());
                                    assert!(light_pdf >= 0.0);
                                    let weight = mis_weight(bsdf_pdf, light_pdf);
                                    li += beta * light.le(&ray) * weight;
                                    // println!("{} {} {}", light_pdf, bsdf_pdf, weight);
                                }
                            }

                            let wo = -ray.d;
                            if depth >= self.max_depth {
                                break;
                            }
                            depth += 1;
                            {
                                let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
                                let sample_self =
                                    if let Some(light2) = scene.get_light_of_shape(shape) {
                                        if light as *const dyn Light == light2 as *const dyn Light {
                                            true
                                        } else {
                                            false
                                        }
                                    } else {
                                        false
                                    };
                                if !sample_self {
                                    let p_ref = ReferencePoint { p, n: ng };
                                    let light_sample = light.sample_li(&sampler.next3d(), &p_ref);
                                    let light_pdf = light_sample.pdf * light_pdf;
                                    if !light_sample.li.is_black()
                                        && !scene.shape.occlude(&light_sample.shadow_ray)
                                    {
                                        let bsdf_pdf = bsdf.evaluate_pdf(&wo, &light_sample.wi);
                                        let weight = mis_weight(light_pdf, bsdf_pdf);
                                        // println!("{} {} {}", light_pdf, bsdf_pdf, weight);
                                        // assert!(light_pdf.is_finite());
                                        assert!(light_pdf >= 0.0);
                                        li += beta
                                            * bsdf.evaluate(&wo, &light_sample.wi)
                                            * glm::dot(&ng, &light_sample.wi).abs()
                                            * light_sample.li
                                            / light_pdf
                                            * weight;
                                    }
                                }
                            }

                            if let Some(bsdf_sample) = bsdf.sample(&sampler.next2d(), &wo) {
                                let wi = &bsdf_sample.wi;
                                ray = Ray::spawn(&p, wi).offset_along_normal(&ng);
                                beta *= bsdf_sample.f * glm::dot(wi, &ng).abs() / bsdf_sample.pdf;
                                prev_bsdf_pdf = Some(bsdf_sample.pdf);
                                prev_n = Some(isct.ng);
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
                acc_li += li;
            }
            acc_li = acc_li / (self.spp as Float);
            film.add_sample(&uvec2(x, y), &acc_li, 1.0);
        });
        film
    }
}