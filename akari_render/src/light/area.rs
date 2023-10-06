use crate::{
    geometry::*,
    interaction::*,
    light::*,
    sampling::uniform_sample_triangle,
    svm::{surface::SURFACE_EVAL_EMISSION, ShaderRef},
    util::distribution::{AliasTableEntry, BindlessAliasTableVar},
};
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct AreaLight {
    pub light_id: u32,
    pub instance_id: u32,
    pub surface: ShaderRef,
    pub area_sampling_index: u32,
}

impl AreaLightExpr {
    fn emission(
        &self,
        wo: Expr<Float3>,
        si: Expr<SurfaceInteraction>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        let emission = Color::from_flat(
            ctx.color_repr(),
            ctx.surface_eval
                .evaluate_ex(
                    self.surface,
                    si,
                    wo,
                    Expr::<Float3>::zeroed(),
                    swl,
                    SURFACE_EVAL_EMISSION.expr(),
                )
                .emission,
        );
        emission
    }
}
impl Light for AreaLightExpr {
    fn id(&self) -> Expr<u32> {
        self.light_id
    }
    fn le(
        &self,
        ray: Expr<Ray>,
        si: Expr<SurfaceInteraction>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        let emission = self.emission(-ray.d, si, swl, ctx);
        let ns = si.geometry.ns;
        select(
            ns.dot(ray.d).lt(0.0),
            emission,
            Color::zero(ctx.color_repr()),
        )
    }
    #[tracked]
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
        let at_entries = area_samplers.buffer::<AliasTableEntry>(self.area_sampling_index);
        let at_pdf = area_samplers.buffer::<f32>(self.area_sampling_index + 1);
        let at = BindlessAliasTableVar(at_entries, at_pdf);
        let (prim_id, pdf, _) = at.sample_and_remap(u_select);
        // let (prim_id, pdf) = (0u32.expr(), 1.0f32.expr());
        let shading_triangle = meshes.shading_triangle(self.instance_id, prim_id);
        let bary = uniform_sample_triangle(u_sample);
        let area = shading_triangle.area();
        let p = shading_triangle.p(bary);
        let uv = shading_triangle.uv(bary);
        let frame = shading_triangle.ortho_frame(bary);
        let n = frame.n;
        let geometry = SurfaceLocalGeometry::from_comps_expr(SurfaceLocalGeometryComps {
            p,
            ng: shading_triangle.ng,
            ns: n,
            tangent: frame.t,
            bitangent: frame.s,
            uv,
        });

        let si = SurfaceInteraction::from_comps_expr(SurfaceInteractionComps {
            inst_id: self.instance_id,
            prim_id,
            bary,
            geometry,
            frame,
            valid: true.expr(),
        });
        let wi = p - pn.p;
        if wi.length_squared() == 0.0 {
            LightSample {
                li: Color::zero(ctx.color_repr()),
                pdf,
                shadow_ray: Expr::<Ray>::zeroed(),
                wi,
                n,
                valid: false.expr(),
            }
        } else {
            let emission = self.emission(-wi, si, swl, ctx);
            let dist2 = wi.length_squared();
            let wi = wi / dist2.sqrt();
            let li = select(wi.dot(n).lt(0.0), emission, Color::zero(ctx.color_repr()));
            let cos_theta_i = n.dot(wi).abs();
            let pdf = pdf / area * dist2 / cos_theta_i;
            let ro = rtx::offset_ray_origin(pn.p, face_forward(pn.n, wi));
            let dist = (p - ro).length();
            let shadow_ray = Ray::new_expr(
                ro,
                wi,
                0.0,
                dist * (1.0f32 - 1e-3),
                Uint2::expr(u32::MAX, u32::MAX),
                Uint2::expr(self.instance_id, prim_id),
            );
            // cpu_dbg!( u);
            LightSample {
                li,
                pdf,
                shadow_ray,
                wi,
                n,
                valid: pdf.is_finite(),
            }
        }
    }
    #[tracked]
    fn pdf_direct(
        &self,
        si: Expr<SurfaceInteraction>,
        pn: Expr<PointNormal>,
        ctx: &LightEvalContext<'_>,
    ) -> Expr<f32> {
        let prim_id = si.prim_id;
        let meshes = ctx.meshes;
        let area_samplers = meshes.mesh_area_samplers.var();
        let at_entries = area_samplers.buffer::<AliasTableEntry>(self.area_sampling_index);
        let at_pdf = area_samplers.buffer::<f32>(self.area_sampling_index + 1);
        let at = BindlessAliasTableVar(at_entries, at_pdf);
        if debug_mode() {
            lc_assert!(si.inst_id.eq(self.instance_id));
        }
        let shading_triangle = meshes.shading_triangle(si.inst_id, si.prim_id);
        let area = shading_triangle.area();
        let prim_pdf = at.pdf(prim_id);
        let bary = si.bary;
        let n = shading_triangle.n(bary);
        let p = shading_triangle.p(bary);
        let wi = p - pn.p;
        let dist2 = wi.length_squared();
        let wi = wi / dist2.sqrt();
        let pdf = prim_pdf / area * dist2 / n.dot(-wi).max_(1e-6);
        pdf
    }
}
impl_polymorphic!(Light, AreaLight);
