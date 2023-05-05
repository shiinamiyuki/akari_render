use crate::{geometry::*, light::*, *};
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct AreaLight {
    pub instance_id: u32,
    pub area_sampling_index: u32,
}

impl Light for AreaLightExpr {
    fn le(&self, ray: Expr<Ray>, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Color {
        let scene = ctx.scene;
        let instances = scene.meshes.mesh_instances.var();
        let instance = instances.read(self.instance_id());
        let emission = ctx.texture(instance.emission_tex());
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
        u: Expr<Float2>,
        ctx: &ShadingContext<'_>,
    ) -> LightSample {
        todo!()
    }
}
impl_polymorphic!(Light, AreaLight);
