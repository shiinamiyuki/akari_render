use crate::{color::*, geometry::*, sampler::*, *};

pub trait Camera {
    fn generate_ray(
        &self,
        pixel: Expr<Uint2>,
        sampler: &dyn Sampler,
        color_repr: &ColorRepr,
    ) -> (Expr<Ray>, Color);
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct PerspectiveCamera {
    pub resolution: Uint2,
    pub c2w: AffineTransform,
    pub w2c: AffineTransform,
    pub fov: f32,
    pub r2c: AffineTransform,
    pub c2r: AffineTransform,
    pub lens_area: f32,
    pub lens_radius: f32,
    pub focal_length: f32,
}
impl PerspectiveCamera {
    pub fn new(
        device: Device,
        resolution: Uint2,
        transform: AffineTransform,
        fov: f32,
        lens_radius: f32,
        focal_length: f32,
    ) -> Buffer<Self> {
        let fov = fov.to_radians();
        let mut m = glam::Mat4::IDENTITY;
        let fres = glam::vec2(resolution.x as f32, resolution.y as f32);
        m = glam::Mat4::from_scale(glam::vec3(1.0 / fres.x, 1.0 / fres.y, 1.0)) * m;
        m = glam::Mat4::from_scale(glam::vec3(2.0, 2.0, 1.0)) * m;
        m = glam::Mat4::from_translation(glam::vec3(-1.0, -1.0, 0.0)) * m;
        m = glam::Mat4::from_scale(glam::vec3(1.0, -1.0, 1.0)) * m;
        let s = (fov / 2.0).tan();
        if resolution.x > resolution.y {
            m = glam::Mat4::from_scale(glam::vec3(s, s * fres.y / fres.x, 1.0)) * m;
        } else {
            m = glam::Mat4::from_scale(glam::vec3(s * fres.x / fres.y, s, 1.0)) * m;
        }
        m = glam::Mat4::from_translation(glam::vec3(0.0, 0.0, -1.0)) * m;
        let r2c = AffineTransform::from_matrix(&m);
        let camera = device.create_buffer::<PerspectiveCamera>(1).unwrap();
        camera.view(..).copy_from(&[Self {
            resolution,
            c2w: transform,
            w2c: transform.inverse(),
            fov,
            r2c,
            c2r: r2c.inverse(),
            lens_area: std::f32::consts::PI * lens_radius * lens_radius,
            lens_radius,
            focal_length,
        }]);
        device
            .create_kernel::<()>(&|| todo!())
            .unwrap()
            .dispatch([1, 1, 1])
            .unwrap();
        camera
    }
}

impl Camera for PerspectiveCameraExpr {
    fn generate_ray(
        &self,
        pixel: Expr<Uint2>,
        sampler: &dyn Sampler,
        color_repr: &ColorRepr,
    ) -> (Expr<Ray>, Color) {
        let fpixel = pixel.float() + make_float2(0.5, 0.5);
        let p_film = fpixel + sampler.next_2d();
        let mut ray = RayExpr::new(
            make_float3(0.0, 0.0, 0.0),
            self.r2c()
                .transform_point(make_float3(p_film.x(), p_film.y(), 0.0))
                .normalize(),
            Float::from(0.0),
            Float::from(1e20),
        );
        ray = ray.set_o(self.c2w().transform_point(ray.o()));
        ray = ray.set_d(self.c2w().transform_vector(ray.d()));
        (ray, Color::one(&color_repr))
    }
}
