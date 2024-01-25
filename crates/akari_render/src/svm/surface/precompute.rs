use crate::{
    color::sample_wavelengths, geometry::FrameExpr, microfacet::TrowbridgeReitzDistribution,
    sampler::Sampler,
};

use super::*;
// Taken from Cycles
#[tracked(crate = "luisa")]
#[allow(dead_code)]
fn precompute_ggx_schlick_s(
    roughness: Expr<f32>,
    mu: Expr<f32>,
    _z: Expr<f32>,
    exponent: Expr<f32>,
    rng: &IndependentSampler,
    color_repr: ColorRepr,
) -> Expr<f32> {
    let f0 = Color::from_flat(color_repr, Float4::expr(0.0, 1.0, 0.0, 0.0));
    let f90 = Color::from_flat(color_repr, Float4::expr(1.0, 1.0, 0.0, 0.0));
    let bsdf = MicrofacetReflection {
        color: Color::one(color_repr),
        fresnel: Box::new(FresnelGeneralizedSchlick { f0, f90, exponent }),
        dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
            Float2::expr(roughness, roughness),
            true,
        )),
    };
    let closure = SurfaceClosure {
        inner: Rc::new(bsdf),
        frame: FrameExpr::from_n(Float3::expr(0.0, 0.0, 1.0)),
        ng: Float3::expr(0.0, 0.0, 1.0),
    };
    let wo = Float3::expr((1.0 - mu.sqr()).sqrt(), 0.0, mu);
    let swl = sample_wavelengths(color_repr, rng).var();
    let sample = closure.sample(
        wo,
        rng.next_1d(),
        rng.next_2d(),
        swl,
        &BsdfEvalContext {
            color_repr,
            ad_mode: ADMode::None,
        },
    );
    if sample.valid & (sample.pdf > 0.0) {
        let f = sample.color.flatten();
        /* The idea here is that the resulting Fresnel factor is always bounded by
         * F0..F90, so it's enough to precompute and store the interpolation factor. */
        (f.x / f.y).clamp(0.0f32.expr(), 1.0f32.expr())
    } else {
        0.0f32.expr()
    }
}
#[tracked(crate = "luisa")]
#[allow(dead_code)]
pub fn precompute_ggx_dielectric(
    roughness: Expr<f32>,
    mu: Expr<f32>,
    ior: Expr<f32>,
    color_repr: ColorRepr,
    rng: &IndependentSampler,
) -> Expr<f32> {
    let bsdf = MicrofacetReflection {
        color: Color::one(color_repr),
        fresnel: Box::new(FresnelDielectric { eta: ior }),
        dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
            Float2::expr(roughness, roughness),
            true,
        )),
    };
    let closure = SurfaceClosure {
        inner: Rc::new(bsdf),
        frame: FrameExpr::from_n(Float3::expr(0.0, 0.0, 1.0)),
        ng: Float3::expr(0.0, 0.0, 1.0),
    };
    let wo = Float3::expr((1.0 - mu.sqr()).sqrt(), 0.0, mu);
    let swl = sample_wavelengths(color_repr, rng).var();
    let sample = closure.sample(
        wo,
        rng.next_1d(),
        rng.next_2d(),
        swl,
        &BsdfEvalContext {
            color_repr,
            ad_mode: ADMode::None,
        },
    );
    if sample.valid & (sample.pdf > 0.0) {
        let f = sample.color.flatten();
        f.x / sample.pdf
    } else {
        0.0f32.expr()
    }
}
impl PreComputedTables {
    pub fn init(device: Device, heap: Arc<MegaHeap>, color_repr: ColorRepr) -> Self {
        let mut tables = Self::new(device, heap);
        // tables.get_or_compute(
        //     "ggx_gen_schlick_ior_s",
        //     PreComputeOptions {
        //         samples: 1 << 20,
        //         dim: [16, 16, 16],
        //         f: track!(|v: Expr<Float3>, rng: &IndependentSampler| {
        //             let roughness = v.x;
        //             let mu = v.y;
        //             let z = v.z;
        //             let ior = ior_parametrization(z);
        //             precompute_ggx_schlick_s(roughness, mu, ior, -1.0f32.expr(), rng, color_repr)
        //         }),
        //     },
        // // );
        // tables.get_or_compute(
        //     &format!("ggx_gen_schlick_s.{}", color_repr.to_string()),
        //     PreComputeOptions {
        //         samples: 1 << 20,
        //         dim: [16, 16, 16],
        //         f: track!(|v: Expr<Float3>, rng: &IndependentSampler| {
        //             let roughness = v.x;
        //             let mu = v.y;
        //             let z = v.z;
        //             let exponent = 5.0 * ((1.0 - z) / z);
        //             precompute_ggx_schlick_s(
        //                 roughness,
        //                 mu,
        //                 1.0f32.expr(),
        //                 exponent,
        //                 rng,
        //                 color_repr,
        //             )
        //         }),
        //     },
        // );
        tables.get_or_compute(
            &format!("ggx_dielectric_s.{}", color_repr.to_string()),
            PreComputeOptions {
                samples: 1 << 20,
                dim: [16, 16, 16],
                f: track!(|v: Expr<Float3>, rng: &IndependentSampler| {
                    let roughness = v.x;
                    let mu = v.y;
                    let ior = ior_parametrization(v.z);
                    precompute_ggx_dielectric(roughness, mu, ior, color_repr, rng)
                }),
            },
        );
        tables
    }
}
