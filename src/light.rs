use crate::distribution::Distribution1D;
use crate::shape::Shape;
use crate::texture::ShadingPoint;
use crate::texture::Texture;
use crate::*;
#[derive(Clone, Copy)]
pub struct LightRaySample {
    pub le: Spectrum,
    pub pdf_dir: Float,
    pub pdf_pos: Float,
    pub ray: Ray,
    pub n: Vec3,
}
#[derive(Clone, Copy)]
pub struct LightSample {
    pub li: Spectrum,
    pub pdf: Float,
    pub shadow_ray: Ray,
    pub wi: Vec3,
    pub p: Vec3,
}
#[derive(Clone, Copy)]
pub struct ReferencePoint {
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
pub trait Light: Sync + Send +AsAny {
    fn sample_le(&self, u: &[Vec2; 2]) -> LightRaySample;
    fn sample_li(&self, u: &Vec3, p: &ReferencePoint) -> LightSample;
    // (pdf_pos,pdf_dir)
    fn pdf_le(&self, ray: &Ray) -> (Float, Float);
    // (pdf_pos,pdf_dir)
    fn pdf_li(&self, wi: &Vec3, p: &ReferencePoint) -> (Float, Float);
    fn le(&self, ray: &Ray) -> Spectrum;
    fn flags(&self) -> LightFlags;
    fn power(&self) -> Float;
    fn address(&self) -> usize; // ????
}
pub trait LightDistribution: Sync + Send + AsAny {
    fn sample<'a>(&'a self, u: Float) -> (&'a dyn Light, Float);
    fn pdf<'a>(&self, light: &'a dyn Light) -> Float;
}
pub struct PowerLightDistribution {
    lights: Vec<Arc<dyn Light>>,
    dist: Distribution1D,
    pdf_map: HashMap<usize, Float>,
}
impl PowerLightDistribution {
    pub fn pdf(&self)->&[f32]{
        &self.dist.pmf
    }
    pub fn new(lights: Vec<Arc<dyn Light>>) -> Self {
        let power: Vec<_> = lights.iter().map(|light| light.power()).collect();
        let dist = Distribution1D::new(power.as_slice()).unwrap();
        let mut pdf_map = HashMap::new();

        for (i, light) in lights.iter().enumerate() {
            let pdf = dist.pdf_discrete(i);
            pdf_map.insert(light.address(), pdf);
        }
        Self {
            lights,
            pdf_map,
            dist,
        }
    }
}
impl LightDistribution for PowerLightDistribution {
    fn sample<'a>(&'a self, u: Float) -> (&'a dyn Light, Float) {
        let (idx, pdf) = self.dist.sample_discrete(u);
        (self.lights[idx].as_ref(), pdf)
    }
    fn pdf<'a>(&self, light: &'a dyn Light) -> Float {
        if let Some(pdf) = self.pdf_map.get(&light.address()) {
            *pdf
        } else {
            0.0
        }
    }
}
impl_as_any!(PowerLightDistribution);
pub struct UniformLightDistribution {
    lights: Vec<Arc<dyn Light>>,
    pdf_map: HashMap<usize, Float>,
}
impl_as_any!(UniformLightDistribution);
impl UniformLightDistribution {
    pub fn new(lights: Vec<Arc<dyn Light>>) -> Self {
        let mut pdf_map = HashMap::new();
        let pdf = 1.0 / lights.len() as Float;
        for i in &lights {
            pdf_map.insert(i.address(), pdf);
        }
        Self { lights, pdf_map }
    }
}
impl LightDistribution for UniformLightDistribution {
    fn sample<'a>(&'a self, u: Float) -> (&'a dyn Light, Float) {
        let idx = ((u * self.lights.len() as Float) as usize).min(self.lights.len() - 1);
        let pdf = 1.0 / self.lights.len() as Float;
        (self.lights[idx].as_ref(), pdf)
    }
    fn pdf<'a>(&self, light: &'a dyn Light) -> Float {
        if let Some(pdf) = self.pdf_map.get(&light.address()) {
            *pdf
        } else {
            0.0
        }
    }
}

pub struct AreaLight {
    pub shape: Arc<dyn Shape>,
    pub emission: Arc<dyn Texture>,
}
impl_as_any!(AreaLight);
impl Light for AreaLight {
    fn sample_le(&self, _u: &[Vec2; 2]) -> LightRaySample {
        todo!()
    }

    fn sample_li(&self, u: &Vec3, ref_: &ReferencePoint) -> LightSample {
        let surface_sample = self.shape.sample_surface(u);
        let li = self.emission.evaluate_s(&ShadingPoint {
            texcoord: surface_sample.texcoords,
        });
        let wi = surface_sample.p - ref_.p;
        let dist2 = glm::dot(&wi, &wi);
        let wi = wi / dist2.sqrt();
        let pdf = surface_sample.pdf * dist2 / glm::dot(&wi, &surface_sample.ng).abs();
        let mut ray = Ray::spawn_to(&surface_sample.p, &ref_.p);
        ray.tmax *= 0.997;
        LightSample {
            li,
            pdf,
            wi,
            p: surface_sample.p,
            shadow_ray: ray.offset_along_normal(&surface_sample.ng),
        }
    }

    fn pdf_le(&self, _ray: &Ray) -> (Float, Float) {
        todo!()
    }

    fn pdf_li(&self, wi: &Vec3, ref_: &ReferencePoint) -> (Float, Float) {
        let ray = Ray::spawn(&ref_.p, wi);
        if let Some(isct) = self.shape.intersect(&ray) {
            let pdf_area = 1.0 / self.shape.area();
            let pdf_sa = pdf_area * isct.t * isct.t / glm::dot(&wi, &isct.ng).abs();
            (pdf_area, pdf_sa)
        } else {
            (0.0, 0.0)
        }
    }

    fn le(&self, ray: &Ray) -> Spectrum {
        if let Some(isct) = self.shape.intersect(ray) {
            self.emission
                .evaluate_s(&ShadingPoint::from_intersection(&isct))
        } else {
            Spectrum::zero()
        }
    }

    fn flags(&self) -> LightFlags {
        LightFlags::NONE
    }

    fn address(&self) -> usize {
        self as *const Self as usize
    }
    fn power(&self) -> Float {
        self.emission.power() * self.shape.area()
    }
}

pub struct PointLight {
    pub position: Vec3,
    pub emission: Arc<dyn Texture>,
}
impl_as_any!(PointLight);
impl PointLight {
    fn evaluate(&self, w: &Vec3) -> Spectrum {
        let uv = spherical_to_uv(&dir_to_spherical(w));
        let sp = ShadingPoint { texcoord: uv };
        self.emission.evaluate_s(&sp)
    }
}
impl Light for PointLight {
    fn sample_le(&self, u: &[Vec2; 2]) -> LightRaySample {
        let w = uniform_sphere_sampling(&u[0]);
        LightRaySample {
            le: self.evaluate(&w),
            pdf_pos: 1.0,
            pdf_dir: uniform_sphere_pdf(),
            ray: Ray::spawn(&self.position, &w),
            n: w,
        }
    }
    fn sample_li(&self, _: &Vec3, ref_: &ReferencePoint) -> LightSample {
        let mut ray = Ray::spawn_to(&self.position, &ref_.p);
        let len2 = {
            let v = self.position - ref_.p;
            glm::dot(&v, &v)
        };
        let wi = glm::normalize(&(self.position - ref_.p));
        ray.tmax *= 0.997;
        LightSample {
            li: self.evaluate(&(-wi)) / len2,
            pdf: 1.0,
            shadow_ray: ray,
            wi,
            p: self.position,
        }
    }
    fn pdf_le(&self, _ray: &Ray) -> (Float, Float) {
        (0.0, uniform_sphere_pdf())
    }
    fn pdf_li(&self, _wi: &Vec3, _p: &ReferencePoint) -> (Float, Float) {
        (0.0, 0.0)
    }
    fn le(&self, _: &Ray) -> Spectrum {
        Spectrum::zero()
        // unimplemented!("point light cannot be hit")
    }
    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_POSITION
    }
    fn address(&self) -> usize {
        self as *const PointLight as usize
    }
    fn power(&self) -> Float {
        self.emission.power()
    }
}