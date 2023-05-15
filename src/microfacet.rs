use crate::geometry::Frame;
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
}

pub struct TrowbridgeReitzDistribution {
    pub alpha: Expr<Float2>,
}
impl TrowbridgeReitzDistribution {
    pub const MIN_ROUGHNESS: f32 = 1e-4;
    pub fn from_alpha(alpha: Expr<Float2>) -> Self {
        Self { alpha }
    }
    pub fn from_roughness(roughness: Expr<Float2>) -> Self {
        let alpha = roughness.sqr().max(Self::MIN_ROUGHNESS);
        Self::from_alpha(alpha)
    }
}
lazy_static! {
    static ref TR_D_IMPL: Callable<(Expr<Float3>, Expr<Float2>), Expr<f32>> =
        create_static_callable::<(Expr<Float3>, Expr<Float2>), Expr<f32>>(|wh, alpha| {
            let tan2_theta = Frame::tan2_theta(wh);
            let cos4_theta = Frame::cos2_theta(wh) * Frame::cos2_theta(wh);
            let ax = alpha.x();
            let ay = alpha.y();
            let e =
                tan2_theta * ((Frame::cos_phi(wh) / ax).sqr() + (Frame::sin_phi(wh) / ay).sqr());
            let d = 1.0 / (PI * ax * ay * cos4_theta * (1.0 + e).sqr());
            select(tan2_theta.is_infinite(), const_(0.0f32), d)
        });
    static ref TR_LAMBDA_IMPL: Callable<(Expr<Float3>, Expr<Float2>), Expr<f32>> =
        create_static_callable::<(Expr<Float3>, Expr<Float2>), Expr<f32>>(|w, alpha| {
            let abs_tan_theta = Frame::tan_theta(w).abs();
            let alpha2 =
                Frame::cos2_phi(w) * alpha.x().sqr() + Frame::sin2_phi(w) * alpha.y().sqr();
            let alpha2_tan2_theta = alpha2 * abs_tan_theta.sqr();
            let l = (-1.0 + (1.0 + alpha2_tan2_theta).sqrt()) * 0.5;
            select(abs_tan_theta.is_infinite(), const_(0.0f32), l)
        });
    static ref TR_SAMPLE_11: Callable<(Expr<f32>, Expr<Float2>), Expr<Float2>> =
        create_static_callable::<(Expr<f32>, Expr<Float2>), Expr<Float2>>(|cos_theta, u| {
            if_then_else(
                cos_theta.cmplt(0.99999),
                || {
                    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
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
                || {
                    let r = (u.x() / (1.0 - u.x())).sqrt();
                    let phi = 2.0 * PI * u.y();
                    make_float2(r * phi.cos(), r * phi.sin())
                },
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
        let s = select(
            Frame::cos_theta(wo).cmpgt(0.0),
            const_(1.0f32),
            const_(-1.0f32),
        );
        let wh = TR_SAMPLE.call(s * wo, self.alpha, u);
        s * wh
    }

    fn pdf(&self, wo: Expr<Float3>, wh: Expr<Float3>) -> Expr<f32> {
        self.d(wh) * self.g1(wo) * wo.dot(wh).abs() / Frame::abs_cos_theta(wo)
    }
}
