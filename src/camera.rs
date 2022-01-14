use crate::*;
use sampler::*;
pub trait Camera: Sync + Send + Base {
    fn generate_ray(&self, pixel: UVec2, sampler: &mut dyn Sampler) -> (Ray, Spectrum);
    fn resolution(&self) -> UVec2;
    fn we(&self, ray: &Ray) -> (Option<UVec2>, Spectrum);
    fn pdf_we(&self, ray: &Ray) -> (f32, f32);
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
        let r2c = Transform::from_matrix(&m);
        let a = {
            let p_min = r2c.transform_point(vec3(0.0, 0.0, 0.0));
            let p_max = r2c.transform_point(vec3(resolution.x as f32, resolution.y as f32, 0.0));
            ((p_max.x - p_min.x) * (p_max.y * p_min.y)).abs()
        };
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
        let p = {
            let v = self.r2c.transform_point(vec3(p_film.x, p_film.y, 0.0));
            vec2(v.x, v.y)
        };
        let mut ray = Ray::spawn(
            Vec3::ZERO,
            (vec3(p.x, p.y, 0.0) - vec3(0.0, 0.0, 1.0)).normalize(),
        );
        // ray.tmin = (1.0 / ray.d.z).abs();
        ray.o = self.c2w.transform_point(ray.o);
        ray.d = self.c2w.transform_vector(ray.d);
        (ray, Spectrum::one())
    }
    fn resolution(&self) -> UVec2 {
        self.resolution
    }
    fn we(&self, ray: &Ray) -> (Option<UVec2>, Spectrum) {
        let cos_theta = ray.d.dot(self.c2w.transform_vector(vec3(0.0, 0.0, -1.0)));
        if cos_theta <= 0.0 {
            return (None, Spectrum::zero());
        }
        let p_focus = ray.at(1.0 / cos_theta);
        let p_raster = self
            .c2r
            .transform_point(self.w2c.transform_point(p_focus));
        if p_raster.x < 0.0
            || p_raster.x >= self.resolution().x as f32
            || p_raster.y < 0.0
            || p_raster.y >= self.resolution().y as f32
        {
            return (None, Spectrum::zero());
        }
        let lens_area = 1.0;
        let cos2_theta = cos_theta * cos_theta;
        (
            Some(uvec2(p_raster.x as u32, p_raster.y as u32)),
            Spectrum::one() / (self.a * lens_area * cos2_theta * cos2_theta),
        )
    }
    fn pdf_we(&self, ray: &Ray) -> (f32, f32) {
        let cos_theta = ray.d.dot(self.c2w.transform_vector(vec3(0.0, 0.0, -1.0)));
        if cos_theta <= 0.0 {
            return (0.0, 0.0);
        }
        let p_focus = ray.at(1.0 / cos_theta);
        let p_raster = self
            .c2r
            .transform_point(self.w2c.transform_point(p_focus));
        if p_raster.x < 0.0
            || p_raster.x >= self.resolution().x as f32
            || p_raster.y < 0.0
            || p_raster.y >= self.resolution().y as f32
        {
            return (0.0, 0.0);
        }
        let lens_area = 1.0;
        let cos2_theta = cos_theta * cos_theta;
        (1.0 / lens_area, self.a * cos2_theta * cos_theta)
    }
    fn n(&self) -> Vec3 {
        self.c2w.transform_vector(vec3(0.0, 0.0, -1.0))
    }
}
