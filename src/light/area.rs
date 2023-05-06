use crate::{
    geometry::*,
    light::*,
    interaction::*,
    sampling::uniform_sample_triangle,
    util::alias_table::{AliasTableEntry, BindlessAliasTableVar},
    *,
};
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct AreaLight {
    pub instance_id: u32,
    pub emission:TagIndex,
    pub area_sampling_index: u32,
}

impl Light for AreaLightExpr {
    fn le(&self, ray: Expr<Ray>, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Color {
        let scene = ctx.scene;
        let emission = ctx.texture(self.emission());
        let emission = emission.dispatch(|tag, _key, tex| tex.evaluate(si, ctx));
        let emission = ctx.color_from_float4(emission);
        let ns = si.geometry().ns();
        select(
            ns.dot(ray.d()).cmplt(0.0),
            emission,
            Color::zero(&ctx.color_repr),
        )
    }

    fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u: Expr<Float4>,
        ctx: &ShadingContext<'_>,
    ) -> LightSample {
        let scene = ctx.scene;
        let meshes = &scene.meshes;
        let area_samplers = meshes.mesh_area_samplers.var();
        let at_entries = area_samplers.buffer::<AliasTableEntry>(self.area_sampling_index());
        let at_pdf = area_samplers.buffer::<f32>(self.area_sampling_index() + 1);
        let at = BindlessAliasTableVar(at_entries, at_pdf);
        let (prim_id, pdf) = at.sample(u.xy());
        let shading_triangle = meshes.shading_triangle(self.instance_id(), prim_id);
        let bary = uniform_sample_triangle(u.zw());
        let area = shading_triangle.area();
        let p = shading_triangle.p(bary);
        let n = shading_triangle.n(bary);
        let uv = shading_triangle.tc(bary);
        let geometry = SurfaceLocalGeometryExpr::new(
            p,
            shading_triangle.ng(),
            n,
            uv,
            Float3Expr::zero(),
            Float3Expr::zero(),
        );

        let si = SurfaceInteractionExpr::new(
            geometry,
            bary,
            prim_id,
            self.instance_id(),
            FrameExpr::from_n(n),
            shading_triangle,
            Bool::from(true),
        );
        let emission = ctx.texture(self.emission());
        let emission = emission.dispatch(|tag, _key, tex| tex.evaluate(si, ctx));
        let emission = ctx.color_from_float4(emission);
        let wi = p - pn.p();
        let dist2 = wi.length_squared();
        let wi = wi / dist2.sqrt();
        let li = select(wi.dot(n).cmplt(0.0), emission, Color::zero(&ctx.color_repr));
        let pdf = pdf / area * dist2 / n.dot(-wi).max(1e-6);
        let ro = rtx::offset_ray_origin(pn.p(), pn.n());
        let dist = (p - ro).length();
        let shadow_ray = RayExpr::new(ro, wi, 1e-3, dist * 0.997);
        LightSample {
            li,
            pdf,
            shadow_ray,
            n,
        }
    }
}
impl_polymorphic!(Light, AreaLight);
