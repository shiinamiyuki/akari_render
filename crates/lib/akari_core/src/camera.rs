use crate::*;
use sampler::*;
#[derive(Clone, Copy)]
pub struct CameraSample {
    pub p: Vec3,
    pub wi: Vec3,
    pub pdf: f32,
    pub n: Vec3,
    pub raster: UVec2,
    pub ray: Ray,
    pub vis_ray: Ray,
    pub we: Spectrum,
}
pub trait Camera: Sync + Send + Base {
    fn generate_ray(&self, pixel: UVec2, sampler: &mut dyn Sampler) -> (Ray, Spectrum);
    fn resolution(&self) -> UVec2;
    fn we(&self, ray: &Ray) -> (Option<UVec2>, Spectrum);
    fn pdf_we(&self, ray: &Ray) -> (f32, f32);
    fn sample_wi(&self, u: Vec2, p: &ReferencePoint) -> Option<CameraSample>;
    fn n(&self) -> Vec3;
}

pub struct PerspectiveCamera {
    pub resolution: UVec2,
    pub c2w: Transform,
    pub w2c: Transform,
    pub fov: f32,
    pub r2c: Transform,
    pub c2r: Transform,
    pub a: f32,
}
impl_base!(PerspectiveCamera);
impl PerspectiveCamera {
    pub fn new(resolution: UVec2, transform: &Transform, fov: f32) -> Self {
        let mut m = Mat4::IDENTITY;
        let fres = vec2(resolution.x as f32, resolution.y as f32);
        m = Mat4::from_scale(vec3(1.0 / fres.x, 1.0 / fres.y, 1.0)) * m;
        m = Mat4::from_scale(vec3(2.0, 2.0, 1.0)) * m;
        m = Mat4::from_translation(vec3(-1.0, -1.0, 0.0)) * m;
        m = Mat4::from_scale(vec3(1.0, -1.0, 1.0)) * m;
        let s = (fov / 2.0).atan();
        if resolution.x > resolution.y {
            m = Mat4::from_scale(vec3(s, s * fres.y / fres.x, 1.0)) * m;
        } else {
            m = Mat4::from_scale(vec3(s * fres.x / fres.y, s, 1.0)) * m;
        }
        m = Mat4::from_translation(vec3(0.0, 0.0, -1.0)) * m;
        let r2c = Transform::from_matrix(&m);
        let a = {
            let p_min = r2c.transform_point(vec3(0.0, 0.0, 0.0));
            let p_max = r2c.transform_point(vec3(resolution.x as f32, resolution.y as f32, 0.0));
            let p_min = p_min / p_min.z;
            let p_max = p_max / p_max.z;
            ((p_max.x - p_min.x) * (p_max.y - p_min.y)).abs()
        };
        assert!(a > 0.0);
        Self {
            resolution,
            c2w: *transform,
            w2c: transform.inverse(),
            r2c,
            c2r: r2c.inverse(),
            fov,
            a,
        }
    }
}
impl Camera for PerspectiveCamera {
    fn generate_ray(&self, pixel: UVec2, sampler: &mut dyn Sampler) -> (Ray, Spectrum) {
        // let p_lens = consine_hemisphere_sampling(sampler.next2d())
        let fpixel: Vec2 = pixel.as_vec2();
        let p_film = sampler.next2d() + fpixel;

        let mut ray = Ray::spawn(
            Vec3::ZERO,
            self.r2c
                .transform_point(vec3(p_film.x, p_film.y, 0.0))
                .normalize(),
        );

        // ray.tmin = (1.0 / ray.d.z).abs();
        ray.o = self.c2w.transform_point(ray.o);
        ray.d = self.c2w.transform_vector(ray.d);
        (ray, Spectrum::one())
    }
    fn resolution(&self) -> UVec2 {
        self.resolution
    }
    fn sample_wi(&self, _u: Vec2, ref_: &ReferencePoint) -> Option<CameraSample> {
        // TODO: area lens
        let p_lens = Vec2::ZERO;
        let p_lens_world = self.c2w.transform_point(p_lens.extend(0.0));

        let wi = p_lens_world - ref_.p;
        let dist = wi.length();
        let wi = wi.normalize();

        let lens_area = 1.0f32;
        let n = self.n();
        let pdf = (dist * dist) / (n.dot(wi).abs() * lens_area);
        let ray = Ray::spawn_to(p_lens_world, ref_.p);
        let (raster, we) = self.we(&ray);
        let vis_ray = Ray::spawn_to(ref_.p, p_lens_world).offset_along_normal(ref_.n);
        Some(CameraSample {
            p: p_lens_world,
            wi,
            pdf,
            ray,
            vis_ray,
            raster: raster?,
            we,
            n,
        })
    }
    fn we(&self, ray: &Ray) -> (Option<UVec2>, Spectrum) {
        let cos_theta = ray.d.dot(self.c2w.transform_vector(vec3(0.0, 0.0, -1.0)));
        if cos_theta <= 0.0 {
            return (None, Spectrum::zero());
        }
        let p_focus = ray.at(1.0 / cos_theta);
        let p_raster = self.c2r.transform_point(self.w2c.transform_point(p_focus));
        // assert!(p_raster.z.abs() < 1e-3);
        if p_raster.x < 0.0
            || p_raster.x >= self.resolution().x as f32
            || p_raster.y < 0.0
            || p_raster.y >= self.resolution().y as f32
        {
            return (None, Spectrum::zero());
        }
        let lens_area = 1.0;
        (
            Some(uvec2(p_raster.x as u32, p_raster.y as u32)),
            Spectrum::one() / (self.a * lens_area * cos_theta.powi(4)),
        )
    }
    fn pdf_we(&self, ray: &Ray) -> (f32, f32) {
        let cos_theta = ray.d.dot(self.c2w.transform_vector(vec3(0.0, 0.0, -1.0)));
        if cos_theta <= 0.0 {
            return (0.0, 0.0);
        }
        let p_focus = ray.at(1.0 / cos_theta);
        let p_raster = self.c2r.transform_point(self.w2c.transform_point(p_focus));
        if p_raster.x < 0.0
            || p_raster.x >= self.resolution().x as f32
            || p_raster.y < 0.0
            || p_raster.y >= self.resolution().y as f32
        {
            return (0.0, 0.0);
        }
        let lens_area = 1.0;
        (1.0 / lens_area, 1.0 / (self.a * cos_theta.powi(3)))
    }
    fn n(&self) -> Vec3 {
        self.c2w.transform_normal(vec3(0.0, 0.0, -1.0))
    }
}
