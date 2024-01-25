use std::f32::consts::PI;

use crate::{util::erf_inv, *};
#[tracked(crate = "luisa")]
pub fn uniform_sample_disk(u: Expr<Float2>) -> Expr<Float2> {
    let r = u.x.sqrt();
    let phi = u.y * 2.0 * std::f32::consts::PI;
    Float2::expr(r * phi.cos(), r * phi.sin())
}
#[tracked(crate = "luisa")]
pub fn invert_uniform_sample_disk(p: Expr<Float2>) -> Expr<Float2> {
    let r = p.x.sqr() + p.y.sqr();
    let phi = p.y.atan2(p.x) / (2.0 * std::f32::consts::PI);
    Float2::expr(r, phi.fract())
}
#[tracked(crate = "luisa")]
pub fn cos_sample_hemisphere(u: Expr<Float2>) -> Expr<Float3> {
    let d = uniform_sample_disk(u);
    let z = (1.0 - d.x * d.x - d.y * d.y).max_(0.0).sqrt();
    Float3::expr(d.x, d.y, z)
}
#[tracked(crate = "luisa")]
pub fn invert_cos_sample_hemisphere(p: Expr<Float3>) -> Expr<Float2> {
    let d = Float2::expr(p.x, p.y);
    invert_uniform_sample_disk(d)
}
#[tracked(crate = "luisa")]
pub fn cos_hemisphere_pdf(cos_theta: Expr<f32>) -> Expr<f32> {
    cos_theta * std::f32::consts::FRAC_1_PI
}
#[tracked(crate = "luisa")]
pub fn uniform_sample_triangle(u: Expr<Float2>) -> Expr<Float2> {
    // let su0 = u.x.sqrt();
    // Float2::expr(1.0 - su0, u.y * su0)
    if u.x < u.y {
        let b0 = u.x / 2.0;
        let b1 = u.y - b0;
        Float2::expr(b0, b1)
    } else {
        let b1 = u.y / 2.0;
        let b0 = u.x - b1;
        Float2::expr(b0, b1)
    }
}
#[tracked(crate = "luisa")]
pub fn sample_gaussian(u: Expr<f32>) -> Expr<f32> {
    2.0f32.sqrt() * erf_inv(2.0 * u - 1.0)
}
#[tracked(crate = "luisa")]
pub fn log_gaussian_pdf(x: Expr<f32>, sigma: Expr<f32>) -> Expr<f32> {
    (1.0 / (sigma * (PI * 2.0).sqrt())).ln() + (-0.5 * x.sqr() / sigma.sqr())
}
#[tracked(crate = "luisa")]
pub fn uniform_discrete_choice_and_remap(n: Expr<u32>, u: Expr<f32>) -> (Expr<u32>, Expr<f32>) {
    let i = (u * n.cast_f32()).floor();
    let i = i.cast_i32().clamp(0, n.cast_i32() - 1).cast_u32();
    let remapped = u * n.cast_f32() - i.cast_f32();
    (i, remapped)
}
#[tracked(crate = "luisa")]
pub fn weighted_discrete_choice2_and_remap<A: Aggregate>(
    weight_a: Expr<f32>,
    a: A,
    b: A,
    u: Expr<f32>,
) -> (A, Expr<f32>) {
    let first = u.lt(weight_a);
    let i = select(first, a, b);
    let remapped = select(first, u / weight_a, (u - weight_a) / (1.0 - weight_a));
    (i, remapped)
}
