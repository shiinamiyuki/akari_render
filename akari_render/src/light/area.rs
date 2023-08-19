use crate::{
    geometry::*,
    interaction::*,
    light::*,
    sampling::uniform_sample_triangle,
    util::alias_table::{AliasTableEntry, BindlessAliasTableVar},
};
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct AreaLight {
    pub light_id: u32,
    pub instance_id: u32,
    pub emission: TagIndex,
    pub area_sampling_index: u32,
}

impl Light for AreaLightExpr {
    fn id(&self) -> Expr<u32> {
        self.light_id()
    }
    fn le(
        &self,
        ray: Expr<Ray>,
        si: Expr<SurfaceInteraction>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        let emission = ctx.texture.evaluate_color(self.emission(), si, swl);
        let ns = si.geometry().ns();
        select(
            ns.dot(ray.d()).cmplt(0.0),
            emission,
            Color::zero(ctx.color_repr),
        )
    }

    fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> LightSample {
        let meshes = ctx.meshes;
        let area_samplers = meshes.mesh_area_samplers.var();
        let at_entries = area_samplers.buffer::<AliasTableEntry>(self.area_sampling_index());
        let at_pdf = area_samplers.buffer::<f32>(self.area_sampling_index() + 1);
        let at = BindlessAliasTableVar(at_entries, at_pdf);
        let (prim_id, pdf, _) = at.sample_and_remap(u_select);
        let shading_triangle = meshes.shading_triangle(self.instance_id(), prim_id);
        let bary = uniform_sample_triangle(u_sample);
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
            self.instance_id(),
            prim_id,
            bary,
            geometry,
            FrameExpr::from_n(n),
            Bool::from(true),
        );
        let emission = ctx.texture.evaluate_color(self.emission(), si, swl);
        let wi = p - pn.p();
        let dist2 = wi.length_squared();
        let wi = wi / dist2.sqrt();
        let li = select(wi.dot(n).cmplt(0.0), emission, Color::zero(ctx.color_repr));
        let pdf = pdf / area * dist2 / n.dot(-wi).max(1e-6);
        let ro = rtx::offset_ray_origin(pn.p(), face_forward(pn.n(), wi));
        let dist = (p - ro).length();
        let shadow_ray = RayExpr::new(
            ro,
            wi,
            0.0,
            dist * (1.0 - 1e-3),
            make_uint2(u32::MAX, u32::MAX),
            make_uint2(self.instance_id(), prim_id),
        );
        // cpu_dbg!( u);
        LightSample {
            li,
            pdf,
            shadow_ray,
            wi,
            n,
        }
    }
    fn pdf_direct(
        &self,
        si: Expr<SurfaceInteraction>,
        pn: Expr<PointNormal>,
        ctx: &LightEvalContext<'_>,
    ) -> Expr<f32> {
        let prim_id = si.prim_id();
        let meshes = ctx.meshes;
        let area_samplers = meshes.mesh_area_samplers.var();
        let at_entries = area_samplers.buffer::<AliasTableEntry>(self.area_sampling_index());
        let at_pdf = area_samplers.buffer::<f32>(self.area_sampling_index() + 1);
        let at = BindlessAliasTableVar(at_entries, at_pdf);
        lc_assert!(si.inst_id().cmpeq(self.instance_id()));
        let shading_triangle = meshes.shading_triangle(si.inst_id(), si.prim_id());
        let area = shading_triangle.area();
        let prim_pdf = at.pdf(prim_id);
        let bary = si.bary();
        let n = shading_triangle.n(bary);
        let p = shading_triangle.p(bary);
        let wi = p - pn.p();
        let dist2 = wi.length_squared();
        let wi = wi / dist2.sqrt();
        let pdf = prim_pdf / area * dist2 / n.dot(-wi).max(1e-6);
        pdf
    }
}
impl_polymorphic!(Light, AreaLight);
