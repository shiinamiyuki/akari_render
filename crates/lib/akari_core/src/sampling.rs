use crate::*;

pub trait Warp<Input, Output> {
    fn warp(&self, u: Input) -> Output;
    fn pdf(&self, output: Output) -> f32;
    fn invert(&self, output: Output) -> Option<Input>;
}

pub struct ConcenstricDisk {}
impl Warp<Vec2, Vec2> for ConcenstricDisk {
    fn warp(&self, u: Vec2) -> Vec2 {
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

    fn pdf(&self, output: Vec2) -> f32 {
        1.0 / (2.0 * PI)
    }

    fn invert(&self, output: Vec2) -> Option<Vec2> {
        if output.x == 0.0 && output.y == 0.0 {
            return Some(Vec2::ZERO);
        }
        let r = output.length();
        let cos_theta = output.x / r;
        let mut theta = cos_theta.acos();
        let p = ((theta / FRAC_PI_4) as i32).clamp(0, 7);
        let x_gt_y = p == 0 || p == 3 || p == 4 || p == 7;
        let mut u = Vec2::ZERO;
        if x_gt_y {
            // u.x = r;
            if p == 3 || p == 4 {
                u.x = -r;
                theta -= PI;
            } else {
                u.x = r;
            }
            if p == 7 || p == 3 {
                theta = theta - 2.0 * PI;
            }
            u.y = theta / FRAC_PI_4 * u.x;
        } else {
            // u.y = r;
            if p == 1 || p == 2 {
                u.y = r;
            } else {
                u.y = -r;
                theta -= PI;
            }
            u.x = (FRAC_PI_2 - theta) / FRAC_PI_4 * u.y;
        }
        Some(u * 0.5 + 0.5)
    }
}

pub struct CosineHemisphere {}
impl Warp<Vec2, Vec3> for CosineHemisphere {
    fn warp(&self, u: Vec2) -> Vec3 {
        let uv = ConcenstricDisk {}.warp(u);
        let r = uv.length_squared();
        let h = (1.0 - r).sqrt();
        vec3(uv.x, h, uv.y)
    }

    fn pdf(&self, output: Vec3) -> f32 {
        if output.y < 0.0 {
            0.0
        } else {
            output.y.abs() * FRAC_1_PI
        }
    }

    fn invert(&self, output: Vec3) -> Option<Vec2> {
        if output.y < 0.0 {
            return None;
        }
        let uv = vec2(output.x, output.z);
        ConcenstricDisk {}.invert(uv)
    }
}

pub struct UniformCone {
    pub cos_theta_max: f32,
}
impl Warp<Vec2, Vec3> for UniformCone {
    fn pdf(&self, w: Vec3) -> f32 {
        let cos_theta = w.y;
        if cos_theta >= 0.0 && cos_theta <= self.cos_theta_max {
            1.0 / (2.0 * PI * (1.0 - self.cos_theta_max))
        } else {
            0.0
        }
    }

    fn warp(&self, u: Vec2) -> Vec3 {
        let cos_theta = (1.0f32 - u[0]) + u[0] * self.cos_theta_max;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = u[1] * 2.0 * PI;
        vec3(phi.cos() * sin_theta, cos_theta, phi.sin() * sin_theta)
    }

    fn invert(&self, w: Vec3) -> Option<Vec2> {
        let cos_theta = w.y;
        if cos_theta >= 0.0 && cos_theta <= self.cos_theta_max {
            let phi = w.z.atan2(w.x);
            let u0 = (cos_theta - 1.0) / (self.cos_theta_max - 1.0);
            Some(vec2(u0, phi / (2.0 * PI)))
        } else {
            None
        }
    }
}
pub struct UniformSphere {}
impl Warp<Vec2, Vec3> for UniformSphere {
    fn warp(&self, u: Vec2) -> Vec3 {
        let z = 1.0 - 2.0 * u[0];
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = 2.0 * PI * u[1];
        vec3(r * phi.cos(), z, r * phi.sin())
    }

    fn pdf(&self, output: Vec3) -> f32 {
        1.0 / (4.0 * PI)
    }

    fn invert(&self, w: Vec3) -> Option<Vec2> {
        let z = w.y;
        let u0 = (1.0 - z) * 0.5;
        let phi = w.z.atan2(w.x);
        Some(vec2(u0, phi / (2.0 * PI)))
    }
}
pub struct VisibleWavelenghts {}
impl Warp<f32, f32> for VisibleWavelenghts {
    fn warp(&self, u: f32) -> f32 {
        538.0 - 138.888889 * (0.85691062 - 1.82750197 * u).atanh()
    }

    fn pdf(&self, lambda: f32) -> f32 {
        if lambda >= 360.0 && lambda <= 830.0 {
            0.0039398042f32 / ((0.0072f32 * (lambda - 538f32)).cosh()).powi(2)
        } else {
            0.0
        }
    }

    fn invert(&self, lambda: f32) -> Option<f32> {
        Some((((lambda - 538.0) / -138.888889).tanh() - 0.85691062) / -1.82750197)
    }
}

// pub fn uniform_sample_triangle(u: Vec2) -> Vec2 {
//     let mut uf = (u[0] as f64 * (1u64 << 32) as f64) as u64; // Fixed point
//     let mut cx = 0.0 as f32;
//     let mut cy = 0.0 as f32;
//     let mut w = 0.5 as f32;

//     for _ in 0..16 {
//         let uu = uf >> 30;
//         let flip = (uu & 3) == 0;

//         cy += if (uu & 1) == 0 { 1.0 } else { 0.0 } * w;
//         cx += if (uu & 2) == 0 { 1.0 } else { 0.0 } * w;

//         w *= if flip { -0.5 } else { 0.5 };
//         uf <<= 2;
//     }

//     let b0 = cx + w / 3.0;
//     let b1 = cy + w / 3.0;
//     vec2(b0, b1)
// }

