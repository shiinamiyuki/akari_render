use crate::*;
pub fn uniform_sample_disk(u: Expr<Float2>) -> Expr<Float2> {
    let r = u.x().sqrt();
    let phi = u.y() * 2.0 * std::f32::consts::PI;
    make_float2(r * phi.cos(), r * phi.sin())
}
pub fn cos_sample_hemisphere(u: Expr<Float2>) -> Expr<Float3> {
    let d = uniform_sample_disk(u);
    let z = (1.0 - d.x() * d.x() - d.y() * d.y()).max(0.0).sqrt();
    make_float3(d.x(), z, d.y())
}
pub fn cos_hemisphere_pdf(cos_theta: Float) -> Float {
    cos_theta * std::f32::consts::FRAC_1_PI
}
pub fn uniform_sample_triangle(u: Expr<Float2>) -> Expr<Float2> {
    let su0 = u.x().sqrt();
    make_float2(1.0 - su0, u.y() * su0)
}
