use std::collections::HashMap;
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
    pub n: Vec3,
}
#[derive(Clone, Copy)]
pub struct LightSample {
    pub li: SampledSpectrum,
    pub pdf: f32,
    pub shadow_ray: Ray,
    pub wi: Vec3,
    pub p: Vec3,
    pub n: Vec3,
}

bitflags! {
    pub struct LightFlags : u8 {
        const NONE = 0b0;
        const DELTA_POSITION = 0b1;
        const DELTA_DIRECTION = 0b10;
        const DELTA = Self::DELTA_POSITION.bits | Self::DELTA_DIRECTION.bits;
    }
}
pub trait Light: Sync + Send + Base {
    fn sample_le(&self, u0: Vec3, u1: Vec2, lambda: SampledWavelengths) -> LightRaySample;
    fn sample_li(&self, u: Vec3, p: &ReferencePoint, lambda: SampledWavelengths) -> LightSample;
    // (pdf_pos,pdf_dir)
    fn pdf_le(&self, ray: &Ray, n: Vec3) -> (f32, f32);
    // (pdf_pos,pdf_dir)
    fn pdf_li(&self, wi: Vec3, p: &ReferencePoint) -> (f32, f32);
    fn le(&self, ray: &Ray, lambda: SampledWavelengths) -> SampledSpectrum;
    fn flags(&self) -> LightFlags;
    fn power(&self) -> f32;
    fn address(&self) -> usize; // ????
    fn is_delta(&self) -> bool {
        self.flags().intersects(LightFlags::DELTA)
    }
}
pub trait LightDistribution: Sync + Send + Base {
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
        let dist = Distribution1D::new(power.as_slice()).unwrap();
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
impl_base!(PowerLightDistribution);
pub struct UniformLightDistribution {
    lights: Vec<Arc<dyn Light>>,
    pdf_map: HashMap<usize, f32>,
}
impl_base!(UniformLightDistribution);
impl UniformLightDistribution {
    pub fn new(lights: Vec<Arc<dyn Light>>) -> Self {
        let mut pdf_map = HashMap::new();
        let pdf = 1.0 / lights.len() as f32;
        for i in &lights {
            pdf_map.insert(i.address(), pdf);
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
        if let Some(pdf) = self.pdf_map.get(&light.address()) {
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
impl_base!(AreaLight);
impl AreaLight {
    fn evaluate(&self, sp: &ShadingPoint, lambda: SampledWavelengths) -> SampledSpectrum {
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
    fn sample_le(&self, u0: Vec3, u1: Vec2, lambda: SampledWavelengths) -> LightRaySample {
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
            ray: Ray::spawn(p.p, frame.to_world(dir)),
        }
    }

    fn sample_li(&self, u: Vec3, ref_: &ReferencePoint, lambda: SampledWavelengths) -> LightSample {
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
        let mut ray = Ray::spawn_to(surface_sample.p, ref_.p);
        ray.tmax *= 0.997;
        LightSample {
            li,
            pdf,
            wi,
            p: surface_sample.p,
            shadow_ray: ray.offset_along_normal(surface_sample.ng),
            n: surface_sample.ng,
        }
    }

    fn pdf_le(&self, ray: &Ray, n: Vec3) -> (f32, f32) {
        (1.0 / self.shape.area(), n.dot(ray.d).abs() * FRAC_1_PI)
    }

    fn pdf_li(&self, wi: Vec3, ref_: &ReferencePoint) -> (f32, f32) {
        let ray = Ray::spawn(ref_.p, wi);
        if let Some(hit) = self.shape.intersect(&ray) {
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

    fn le(&self, ray: &Ray, lambda: SampledWavelengths) -> SampledSpectrum {
        if let Some(hit) = self.shape.intersect(ray) {
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

    fn address(&self) -> usize {
        self as *const Self as usize
    }
    fn power(&self) -> f32 {
        self.emission.power() * self.shape.area()
    }
}

pub struct PointLight {
    pub position: Vec3,
    pub emission: Arc<dyn SpectrumTexture>,
    pub colorspace: Option<RgbColorSpace>,
}
impl_base!(PointLight);
impl PointLight {
    fn evaluate(&self, w: Vec3, lambda: SampledWavelengths) -> SampledSpectrum {
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
    fn sample_le(&self, _: Vec3, u1: Vec2, lambda: SampledWavelengths) -> LightRaySample {
        let w = uniform_sphere_sampling(u1);
        LightRaySample {
            le: self.evaluate(w, lambda),
            pdf_pos: 1.0,
            pdf_dir: uniform_sphere_pdf(),
            ray: Ray::spawn(self.position, w),
            n: w,
        }
    }
    fn sample_li(&self, _: Vec3, ref_: &ReferencePoint, lambda: SampledWavelengths) -> LightSample {
        let mut ray = Ray::spawn_to(self.position, ref_.p);
        let len2 = {
            let v = self.position - ref_.p;
            v.length_squared()
        };
        let wi = (self.position - ref_.p).normalize();
        ray.tmax *= 0.997;
        LightSample {
            li: self.evaluate(-wi, lambda) / len2,
            pdf: 1.0,
            shadow_ray: ray,
            wi,
            p: self.position,
            n: ray.d.normalize(),
        }
    }
    fn pdf_le(&self, _ray: &Ray, n: Vec3) -> (f32, f32) {
        (0.0, uniform_sphere_pdf())
    }
    fn pdf_li(&self, _wi: Vec3, _p: &ReferencePoint) -> (f32, f32) {
        (0.0, 0.0)
    }
    fn le(&self, _: &Ray, _lambda: SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::zero()
        // unimplemented!("point light cannot be hit")
    }
    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_POSITION
    }
    fn address(&self) -> usize {
        self as *const PointLight as usize
    }
    fn power(&self) -> f32 {
        self.emission.power()
    }
}
