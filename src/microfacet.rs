use crate::geometry::{face_forward, spherical_to_xyz2, Frame};
use crate::*;
use lazy_static::lazy_static;
use std::f32::consts::PI;

pub trait MicrofacetDistribution {
    fn g1(&self, w: Expr<Float3>) -> Expr<f32> {
        1.0 / (1.0 + self.lambda(w))
    }
    fn g(&self, wo: Expr<Float3>, wi: Expr<Float3>) -> Expr<f32> {
        1.0 / (1.0 + self.lambda(wo) + self.lambda(wi))
    }
    fn d(&self, wh: Expr<Float3>) -> Expr<f32>;
    fn lambda(&self, w: Expr<Float3>) -> Expr<f32>;
    fn sample_wh(&self, wo: Expr<Float3>, u: Expr<Float2>) -> Expr<Float3>;
    fn pdf(&self, wo: Expr<Float3>, wh: Expr<Float3>) -> Expr<f32>;
    fn roughness(&self) -> Expr<f32>;
}

pub struct TrowbridgeReitzDistribution {
    pub alpha: Expr<Float2>,
    pub sample_visible: bool,
    roughness: Expr<f32>,
}
impl TrowbridgeReitzDistribution {
    pub const MIN_ALPHA: f32 = 1e-4;
    pub fn from_alpha(alpha: Expr<Float2>, sample_visible: bool) -> Self {
        Self {
            alpha: alpha.max(Self::MIN_ALPHA),
            sample_visible,
            roughness: (alpha.max(Self::MIN_ALPHA).reduce_sum() * 0.5).sqrt(),
        }
    }
    pub fn from_roughness(roughness: Expr<Float2>, sample_visible: bool) -> Self {
        let alpha = roughness.sqr();
        Self::from_alpha(alpha, sample_visible)
    }
}
lazy_static! {
    static ref TR_D_IMPL: Callable<(Expr<Float3>, Expr<Float2>), Expr<f32>> =
        create_static_callable::<(Expr<Float3>, Expr<Float2>), Expr<f32>>(|wh, alpha| {
            let tan2_theta = Frame::tan2_theta(wh);
            let cos4_theta = Frame::cos2_theta(wh).sqr();
            let ax = alpha.x();
            let ay = alpha.y();
            let e =
                tan2_theta * ((Frame::cos_phi(wh) / ax).sqr() + (Frame::sin_phi(wh) / ay).sqr());
            let d = 1.0 / (PI * ax * ay * cos4_theta * (1.0 + e).sqr());
            select(tan2_theta.is_infinite(), const_(0.0f32), d)

            // Float tan2Theta = Tan2Theta(wm);
            // if (IsInf(tan2Theta))
            //     return 0;
            // Float cos4Theta = Sqr(Cos2Theta(wm));
            // if (cos4Theta < 1e-16f)
            //     return 0;
            // Float e = tan2Theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));
            // return 1 / (Pi * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
        });
    static ref TR_LAMBDA_IMPL: Callable<(Expr<Float3>, Expr<Float2>), Expr<f32>> =
        create_static_callable::<(Expr<Float3>, Expr<Float2>), Expr<f32>>(|w, alpha| {
            let abs_tan_theta = Frame::tan_theta(w).abs();
            let alpha2 =
                Frame::cos2_phi(w) * alpha.x().sqr() + Frame::sin2_phi(w) * alpha.y().sqr();
            let alpha2_tan2_theta = alpha2 * abs_tan_theta.sqr();
            let l = (-1.0 + (1.0 + alpha2_tan2_theta).sqrt()) * 0.5;
            select(!abs_tan_theta.is_finite(), const_(0.0f32), l)
        });
    static ref TR_SAMPLE_11: Callable<(Expr<f32>, Expr<Float2>), Expr<Float2>> =
        create_static_callable::<(Expr<f32>, Expr<Float2>), Expr<Float2>>(|cos_theta, u| {
            if_!(
                cos_theta.cmplt(0.99999),
                 {
                    let sin_theta = (1.0 - cos_theta.sqr()).max(0.0).sqrt();
                    let tan_theta = sin_theta / cos_theta;
                    let a = 1.0 / tan_theta;
                    let g1 = 2.0 / (1.0 + (1.0 + 1.0 / a.sqr()).sqrt());

                    let a = 2.0 * u.x() / g1 - 1.0;
                    let tmp = (1.0 / (a.sqr() - 1.0)).min(1e10f32);
                    let b = tan_theta;
                    let d = ((b * tmp).sqr() - (a.sqr() - b.sqr()) * tmp)
                        .max(0.0)
                        .sqrt();
                    let slope_x_1 = b * tmp - d;
                    let slope_x_2 = b * tmp + d;
                    let slope_x = select(
                        a.cmplt(0.0) | (slope_x_2 * tan_theta).cmpgt(1.0),
                        slope_x_1,
                        slope_x_2,
                    );

                    let s = select(u.y().cmpgt(0.5), const_(1.0f32), const_(-1.0f32));
                    let u2 = select(u.y().cmpgt(0.5), 2.0 * (u.y() - 0.5), 2.0 * (0.5 - u.y()));
                    let z = (u2 * (u2 * (u2 * 0.27385 - 0.73369) + 0.46341))
                        / (u2 * (u2 * (u2 * 0.093073 + 0.309420) - 1.000000) + 0.597999);
                    let slope_y = s * z * (1.0 + slope_x.sqr()).sqrt();
                    make_float2(slope_x, slope_y)
                },
                else {
                    let r = (u.x() / (1.0 - u.x())).sqrt();
                    let phi = 2.0 * PI * u.y();
                    make_float2(r * phi.cos(), r * phi.sin())
                }
            )
        });
    static ref TR_SAMPLE: Callable<(Expr<Float3>, Expr<Float2>, Expr<Float2>), Expr<Float3>> =
        create_static_callable::<(Expr<Float3>, Expr<Float2>, Expr<Float2>), Expr<Float3>>(
            |wi, alpha, u| {
                let wi_stretched =
                    make_float3(alpha.x() * wi.x(), wi.y(), alpha.y() * wi.z()).normalize();
                let slope = TR_SAMPLE_11.call(Frame::cos_theta(wi_stretched), u);

                let slope = make_float2(
                    Frame::cos_phi(wi_stretched) * slope.x()
                        - Frame::sin_phi(wi_stretched) * slope.y(),
                    Frame::sin_phi(wi_stretched) * slope.x()
                        + Frame::cos_phi(wi_stretched) * slope.y(),
                );
                let slope = alpha * slope;
                make_float3(-slope.x(), 1.0, -slope.y()).normalize()
            }
        );
}
impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn d(&self, wh: Expr<Float3>) -> Expr<f32> {
        TR_D_IMPL.call(wh, self.alpha)
    }

    fn lambda(&self, w: Expr<Float3>) -> Expr<f32> {
        TR_LAMBDA_IMPL.call(w, self.alpha)
    }

    fn sample_wh(&self, wo: Expr<Float3>, u: Expr<Float2>) -> Expr<Float3> {
        if self.sample_visible {
            let s = select(
                Frame::cos_theta(wo).cmpgt(0.0),
                const_(1.0f32),
                const_(-1.0f32),
            );
            let wh = TR_SAMPLE.call(s * wo, self.alpha, u);
            s * wh
        } else {
            lazy_static! {
                static ref SAMPLE: Callable<(Expr<Float2>, Expr<Float2>), Expr<Float3>> =
                    create_static_callable::<(Expr<Float2>, Expr<Float2>), Expr<Float3>>(
                        |alpha: Expr<Float2>, u: Expr<Float2>| {
                            let (phi, cos_theta) = if_!(alpha.x().cmpeq(alpha.y()), {
                                let phi = 2.0 * PI * u.y();
                                let tan_theta2 = alpha.x().sqr() * u.x() / (1.0 - u.x());
                                let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
                                (phi, cos_theta)
                            }, else {
                                let phi = (alpha.y() / alpha.x() * (2.0 * PI * u.y() + PI * 0.5).tan()).atan();
                                let phi = select(u.y().cmpgt(0.5), phi + PI, phi);
                                let sin_phi = phi.sin();
                                let cos_phi = phi.cos();
                                let ax2 = alpha.x().sqr();
                                let ay2 = alpha.y().sqr();
                                let a2 = 1.0 / (cos_phi.sqr() / ax2 + sin_phi.sqr() / ay2);
                                let tan_theta2 = a2 * u.x() / (1.0 - u.x());
                                let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
                                (phi, cos_theta)
                            });
                            let sin_theta = (1.0 - cos_theta.sqr()).max(0.0).sqrt();
                            let wh = spherical_to_xyz2(cos_theta, sin_theta, phi);
                            let wh = face_forward(wh, make_float3(0.0, 1.0, 0.0));
                            wh
                        }
                    );
            }
            SAMPLE.call(self.alpha, u)
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wh: Expr<Float3>) -> Expr<f32> {
        if self.sample_visible {
            self.d(wh) * self.g1(wo) * wo.dot(wh).abs() / Frame::abs_cos_theta(wo)
        } else {
            self.d(wh) * Frame::abs_cos_theta(wh)
        }
    }
    fn roughness(&self) -> Expr<f32> {
        self.roughness
    }
}
#[cfg(test)]
mod test {
    use std::env::current_exe;

    use crate::sampler::{init_pcg32_buffer, IndependentSampler, Pcg32, Sampler};

    use super::*;

    #[test]
    fn tr_sample_wh() {
        let ctx = luisa::Context::new(current_exe().unwrap());
        let device = ctx.create_cpu_device();
        let seeds = init_pcg32_buffer(device.clone(), 8192);
        let out = device.create_buffer::<f32>(seeds.len());
        let n_iters = 4098;
        let kernel =
            device.create_kernel::<(Float3, Float2)>(&|wo: Expr<Float3>, alpha: Expr<Float2>| {
                let i = dispatch_id().x();
                let sampler = IndependentSampler {
                    state: var!(Pcg32, seeds.var().read(i)),
                };
                let out = out.var();
                let dist = TrowbridgeReitzDistribution::from_alpha(alpha, false);
                for_range(const_(0)..const_(n_iters), |_| {
                    let wh = dist.sample_wh(wo, sampler.next_2d());
                    let pdf = dist.pdf(wo, wh);
                    if_!(pdf.cmpgt(0.0), {
                        out.write(i, out.read(i) + 1.0 / pdf);
                    });
                });
            });
        let test_alpha = |theta: f32, alpha_x: f32, alpha_y: f32| {
            kernel.dispatch(
                [seeds.len() as u32, 1, 1],
                &Float3::new(theta.sin(), theta.cos(), 0.0),
                &Float2::new(alpha_x, alpha_y),
            );
            let out = out.copy_to_vec();
            let mean =
                out.iter().map(|x| *x as f64).sum::<f64>() / out.len() as f64 / n_iters as f64;
            println!("theta: {}, alpha: {}, mean: {}", theta, alpha_x, mean);
        };
        test_alpha(0.8, 0.1, 0.1);
    }
}