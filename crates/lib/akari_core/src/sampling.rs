use crate::*;

pub fn concentric_sample_disk(u: Vec2) -> Vec2 {
    let u_offset: Vec2 = 2.0 * u - vec2(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return vec2(0.0, 0.0);
    }

    let (theta, r) = {
        if u_offset.x.abs() > u_offset.y.abs() {
            let r = u_offset.x;
            let theta = FRAC_PI_4 * (u_offset.y / u_offset.x);
            (theta, r)
        } else {
            let r = u_offset.y;
            let theta = FRAC_PI_2 - FRAC_PI_4 * (u_offset.x / u_offset.y);
            (theta, r)
        }
    };
    r * vec2(theta.cos(), theta.sin())
}
pub fn consine_hemisphere_sampling(u: Vec2) -> Vec3 {
    let uv = concentric_sample_disk(u);
    let r = uv.length_squared();
    let h = (1.0 - r).sqrt();
    vec3(uv.x, h, uv.y)
}
pub fn uniform_cone_pdf(cos_theta: f32) -> f32 {
    1.0 / (2.0 * PI * (1.0 - cos_theta))
}
pub fn uniform_sample_cone(u: Vec2, cos_theta_max: f32) -> Vec3 {
    let cos_theta = (1.0f32 - u[0]) + u[0] * cos_theta_max;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = u[1] * 2.0 * PI;
    return vec3(phi.cos() * sin_theta, cos_theta, phi.sin() * sin_theta);
}
pub fn uniform_sample_sphere(u: Vec2) -> Vec3 {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    vec3(r * phi.cos(), z, r * phi.sin())
}
pub fn uniform_sphere_pdf() -> f32 {
    1.0 / (4.0 * PI)
}
pub fn uniform_sample_triangle(u: Vec2) -> Vec2 {
    let mut uf = (u[0] as f64 * (1u64 << 32) as f64) as u64; // Fixed point
    let mut cx = 0.0 as f32;
    let mut cy = 0.0 as f32;
    let mut w = 0.5 as f32;

    for _ in 0..16 {
        let uu = uf >> 30;
        let flip = (uu & 3) == 0;

        cy += if (uu & 1) == 0 { 1.0 } else { 0.0 } * w;
        cx += if (uu & 2) == 0 { 1.0 } else { 0.0 } * w;

        w *= if flip { -0.5 } else { 0.5 };
        uf <<= 2;
    }

    let b0 = cx + w / 3.0;
    let b1 = cy + w / 3.0;
    vec2(b0, b1)
}

pub fn visible_wavelenghts_pdf(lambda: f32) -> f32 {
    if lambda >= 360.0 && lambda <= 830.0 {
        0.0039398042f32 / ((0.0072f32 * (lambda - 538f32)).cosh()).powi(2)
    } else {
        0.0
    }
}
pub fn sample_visible_wavelenghts(u: f32) -> f32 {
    538.0 - 138.888889 * (0.85691062 - 1.82750197 * u).atanh()
}
