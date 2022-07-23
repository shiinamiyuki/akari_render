use std::collections::HashMap;
use std::process::exit;
use std::sync::Arc;

use crate::distribution::Distribution1D;
use crate::shape::Shape;
use crate::texture::ShadingPoint;
use crate::texture::SpectrumTexture;
use crate::*;
use bitflags::bitflags;
#[derive(Clone, Copy)]
pub struct LightRaySample {
    pub le: SampledSpectrum,
    pub pdf_dir: f32,
    pub pdf_pos: f32,
    pub ray: Ray,
    pub n: Vec3A,
}
#[derive(Clone, Copy)]
pub struct LightSample {
    pub li: SampledSpectrum,
    pub pdf: f32,
    pub shadow_ray: Ray,
    pub wi: Vec3A,
    pub p: Vec3A,
    pub n: Vec3A,
}

bitflags! {
    pub struct LightFlags : u8 {
        const NONE = 0b0;
        const DELTA_POSITION = 0b1;
        const DELTA_DIRECTION = 0b10;
        const DELTA = Self::DELTA_POSITION.bits | Self::DELTA_DIRECTION.bits;
    }
}

pub trait Light: Sync + Send + AsAny {
    fn sample_emission(&self, u0: Vec3A, u1: Vec2, lambda: &SampledWavelengths) -> LightRaySample;
    fn sample_direct(&self, u: Vec3A, p: &ReferencePoint, lambda: &SampledWavelengths) -> LightSample;
    // (pdf_pos,pdf_dir)
    fn pdf_emission(&self, ray: &Ray, n: Vec3A) -> (f32, f32);
    // (pdf_pos,pdf_dir)
    fn pdf_direct(&self, wi: Vec3A, p: &ReferencePoint) -> (f32, f32);
    fn emission(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum;
    fn flags(&self) -> LightFlags;
    fn power(&self) -> f32;
    fn is_delta(&self) -> bool {
        self.flags().intersects(LightFlags::DELTA)
    }
}
pub trait LightDistribution: Sync + Send + AsAny {
    fn sample<'a>(&'a self, u: f32) -> (&'a dyn Light, f32);
    fn pdf<'a>(&self, light: &'a dyn Light) -> f32;
}
pub struct PowerLightDistribution {
    lights: Vec<Arc<dyn Light>>,
    dist: Distribution1D,
    pdf_map: HashMap<usize, f32>,
}
impl PowerLightDistribution {
    pub fn pdf(&self) -> &[f32] {
        &self.dist.pmf
    }
    pub fn new(lights: Vec<Arc<dyn Light>>) -> Self {
        let power: Vec<_> = lights.iter().map(|light| light.power()).collect();
        let dist = Distribution1D::new(power.as_slice()).unwrap_or_else(|| -> Distribution1D {
            log::error!("no lights defined in scene");
            exit(-1);
        });
        let mut pdf_map = HashMap::new();

        for (i, light) in lights.iter().enumerate() {
            let pdf = dist.pdf_discrete(i);
            pdf_map.insert(Arc::as_ptr(light).cast::<()>() as usize, pdf);
        }
        Self {
            lights,
            pdf_map,
            dist,
        }
    }
}
impl LightDistribution for PowerLightDistribution {
    fn sample<'a>(&'a self, u: f32) -> (&'a dyn Light, f32) {
        let (idx, pdf) = self.dist.sample_discrete(u);
        (self.lights[idx].as_ref(), pdf)
    }
    fn pdf<'a>(&self, light: &'a dyn Light) -> f32 {
        if let Some(pdf) = self
            .pdf_map
            .get(&((light as *const dyn Light).cast::<()>() as usize))
        {
            *pdf
        } else {
            0.0
        }
    }
}

pub struct UniformLightDistribution {
    lights: Vec<Arc<dyn Light>>,
    pdf_map: HashMap<usize, f32>,
}

impl UniformLightDistribution {
    pub fn new(lights: Vec<Arc<dyn Light>>) -> Self {
        let mut pdf_map = HashMap::new();
        let pdf = 1.0 / lights.len() as f32;
        for i in &lights {
            let addr = Arc::as_ptr(i).cast::<()>() as usize;
            pdf_map.insert(addr, pdf);
        }
        Self { lights, pdf_map }
    }
}
impl LightDistribution for UniformLightDistribution {
    fn sample<'a>(&'a self, u: f32) -> (&'a dyn Light, f32) {
        let idx = ((u * self.lights.len() as f32) as usize).min(self.lights.len() - 1);
        let pdf = 1.0 / self.lights.len() as f32;
        (self.lights[idx].as_ref(), pdf)
    }
    fn pdf<'a>(&self, light: &'a dyn Light) -> f32 {
        let addr = (light as *const dyn Light).cast::<()>() as usize;
        if let Some(pdf) = self.pdf_map.get(&addr) {
            *pdf
        } else {
            0.0
        }
    }
}

pub struct AreaLight {
    pub shape: Arc<dyn Shape>,
    pub emission: Arc<dyn SpectrumTexture>,
    pub colorspace: Option<RgbColorSpace>,
}

impl AreaLight {
    fn evaluate(&self, sp: &ShadingPoint, lambda: &SampledWavelengths) -> SampledSpectrum {
        let s = self.emission.evaluate(sp, lambda);
        if let Some(colorspace) = self.colorspace {
            let illuminant = colorspace.illuminant();
            let i = illuminant.sample(lambda);
            s * i
        } else {
            s
        }
    }
}
impl Light for AreaLight {
    fn sample_emission(&self, u0: Vec3A, u1: Vec2, lambda: &SampledWavelengths) -> LightRaySample {
        let p = self.shape.sample_surface(u0);
        let dir = consine_hemisphere_sampling(u1);
        let frame = Frame::from_normal(p.ng);
        LightRaySample {
            le: self.evaluate(
                &ShadingPoint {
                    texcoord: p.texcoords,
                },
                lambda,
            ),
            pdf_dir: (dir.y.abs()) * FRAC_1_PI,
            pdf_pos: p.pdf,
            n: p.ng,
            ray: Ray::spawn(p.p, frame.to_world(dir)).offset_along_normal(p.ng),
        }
    }

    fn sample_direct(
        &self,
        u: Vec3A,
        ref_: &ReferencePoint,
        lambda: &SampledWavelengths,
    ) -> LightSample {
        let surface_sample = self.shape.sample_surface(u);
        let li = self.evaluate(
            &ShadingPoint {
                texcoord: surface_sample.texcoords,
            },
            lambda,
        );
        let wi = surface_sample.p - ref_.p;
        let dist2 = wi.length_squared();
        let wi = wi / dist2.sqrt();
        let cos_theta = -wi.dot(surface_sample.ng);
        let pdf = if cos_theta <= 0.0 {
            0.0
        } else {
            surface_sample.pdf * dist2 / cos_theta
        };
        let mut ray = Ray::spawn_to_offseted1(surface_sample.p, ref_.p, surface_sample.ng);
        ray.tmax *= 1.0 - 1e-3;
        LightSample {
            li,
            pdf,
            wi,
            p: surface_sample.p,
            shadow_ray: ray.offset_along_normal(surface_sample.ng),
            n: surface_sample.ng,
        }
    }

    fn pdf_emission(&self, ray: &Ray, n: Vec3A) -> (f32, f32) {
        (1.0 / self.shape.area(), n.dot(ray.d).abs() * FRAC_1_PI)
    }

    fn pdf_direct(&self, wi: Vec3A, ref_: &ReferencePoint) -> (f32, f32) {
        let ray = Ray::spawn(ref_.p, wi);
        if let Some(hit) = self.shape.intersect(&ray, None) {
            if ray.d.dot(hit.ng) < 0.0 {
                let pdf_area = 1.0 / self.shape.area();
                let pdf_sa = pdf_area * hit.t * hit.t / wi.dot(hit.ng).abs();
                (pdf_area, pdf_sa)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        }
    }

    fn emission(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        if let Some(hit) = self.shape.intersect(ray, None) {
            if hit.ng.dot(ray.d) < 0.0 {
                self.evaluate(&ShadingPoint::from_rayhit(&self.shape, hit), lambda)
            } else {
                SampledSpectrum::zero()
            }
        } else {
            SampledSpectrum::zero()
        }
    }

    fn flags(&self) -> LightFlags {
        LightFlags::NONE
    }

    fn power(&self) -> f32 {
        self.emission.power() * self.shape.area()
    }
}
pub struct SpotLight {
    pub position: Vec3A,
    pub direction: Vec3A,
    pub max_angle: f32,
    pub falloff: f32,
    pub emission: Arc<dyn SpectrumTexture>,
    pub colorspace: Option<RgbColorSpace>,
}
impl SpotLight {
    fn falloff(&self, w: Vec3A) -> f32 {
        let cos = self.direction.dot(w);
        if cos < self.max_angle {
            return 0.0;
        }
        if cos > self.falloff {
            return 1.0;
        }
        let d = (cos - self.max_angle) / (self.falloff - self.max_angle);
        d.powi(4)
    }
    fn evaluate(&self, w: Vec3A, lambda: &SampledWavelengths) -> SampledSpectrum {
        let uv = spherical_to_uv(dir_to_spherical(w));
        let sp = ShadingPoint { texcoord: uv };
        let s = self.emission.evaluate(&sp, lambda);
        let falloff = self.falloff(w);
        if let Some(colorspace) = self.colorspace {
            let illuminant = colorspace.illuminant();
            let i = illuminant.sample(lambda);
            s * i * falloff
        } else {
            s * falloff
        }
    }
}
impl Light for SpotLight {
    fn sample_emission(&self, _u0: Vec3A, u1: Vec2, lambda: &SampledWavelengths) -> LightRaySample {
        let w = uniform_sample_cone(u1, self.max_angle);
        let frame = Frame::from_normal(self.direction);
        let w = frame.to_world(w);
        LightRaySample {
            le: self.evaluate(w, lambda),
            pdf_pos: 1.0,
            pdf_dir: uniform_cone_pdf(self.max_angle),
            ray: Ray::spawn(self.position, w),
            n: w,
        }
    }

    fn sample_direct(
        &self,
        _u: Vec3A,
        ref_: &ReferencePoint,
        lambda: &SampledWavelengths,
    ) -> LightSample {
        let mut ray = Ray::spawn_to(self.position, ref_.p);
        let len2 = {
            let v = self.position - ref_.p;
            v.length_squared()
        };
        let wi = (self.position - ref_.p).normalize();
        ray.tmax *= 1.0 - 1e-3;
        LightSample {
            li: self.evaluate(-wi, lambda) / len2,
            pdf: 1.0,
            shadow_ray: ray,
            wi,
            p: self.position,
            n: ray.d.normalize(),
        }
    }

    fn pdf_emission(&self, _ray: &Ray, _n: Vec3A) -> (f32, f32) {
        (0.0, uniform_cone_pdf(self.max_angle))
    }

    fn pdf_direct(&self, _wi: Vec3A, _p: &ReferencePoint) -> (f32, f32) {
        (0.0, 0.0)
    }

    fn emission(&self, _ray: &Ray, _lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::zero()
    }

    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_POSITION
    }

    fn power(&self) -> f32 {
        self.emission.power() * 2.0 * PI * (1.0 - 0.5 * (self.max_angle + self.falloff))
    }
}
pub struct PointLight {
    pub position: Vec3A,
    pub emission: Arc<dyn SpectrumTexture>,
    pub colorspace: Option<RgbColorSpace>,
}

impl PointLight {
    fn evaluate(&self, w: Vec3A, lambda: &SampledWavelengths) -> SampledSpectrum {
        let uv = spherical_to_uv(dir_to_spherical(w));
        let sp = ShadingPoint { texcoord: uv };
        let s = self.emission.evaluate(&sp, lambda);
        if let Some(colorspace) = self.colorspace {
            let illuminant = colorspace.illuminant();
            let i = illuminant.sample(lambda);
            s * i
        } else {
            s
        }
    }
}
impl Light for PointLight {
    fn sample_emission(&self, _: Vec3A, u1: Vec2, lambda: &SampledWavelengths) -> LightRaySample {
        let w = uniform_sample_sphere(u1);
        LightRaySample {
            le: self.evaluate(w, lambda),
            pdf_pos: 1.0,
            pdf_dir: uniform_sphere_pdf(),
            ray: Ray::spawn(self.position, w),
            n: w,
        }
    }
    fn sample_direct(
        &self,
        _: Vec3A,
        ref_: &ReferencePoint,
        lambda: &SampledWavelengths,
    ) -> LightSample {
        let mut ray = Ray::spawn_to(self.position, ref_.p);
        let len2 = {
            let v = self.position - ref_.p;
            v.length_squared()
        };
        let wi = (self.position - ref_.p).normalize();
        ray.tmax *= 1.0 - 1e-3;
        LightSample {
            li: self.evaluate(-wi, lambda) / len2,
            pdf: 1.0,
            shadow_ray: ray,
            wi,
            p: self.position,
            n: ray.d.normalize(),
        }
    }
    fn pdf_emission(&self, _ray: &Ray, n: Vec3A) -> (f32, f32) {
        (0.0, uniform_sphere_pdf())
    }
    fn pdf_direct(&self, _wi: Vec3A, _p: &ReferencePoint) -> (f32, f32) {
        (0.0, 0.0)
    }
    fn emission(&self, _: &Ray, _lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::zero()
        // unimplemented!("point light cannot be hit")
    }
    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_POSITION
    }

    fn power(&self) -> f32 {
        self.emission.power() * 4.0 * PI
    }
}
