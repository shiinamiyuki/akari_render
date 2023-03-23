use std::{path::PathBuf, rc::Rc};

use crate::{
    camera::Camera,
    geometry::*,
    interaction::*,
    light::{Light, LightDistribution},
    scenegraph::node,
    surface::{Bsdf, Surface},
    texture::Texture,
    *,
};
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
pub struct MeshInstance {
    pub geom_id: u32,
    pub transform: AffineTransform,
    pub normal_index: u32,
    pub texcoord_index: u32,
}
impl MeshInstanceExpr {
    pub fn has_normal(&self) -> Bool {
        self.normal_index().cmpne(u32::MAX)
    }
    pub fn has_texcoord(&self) -> Bool {
        self.texcoord_index().cmpne(u32::MAX)
    }
}

pub struct MeshAggregate {
    pub mesh_vertices: BindlessArray,
    pub mesh_indices: BindlessArray,
    pub mesh_normals: BindlessArray,
    pub mesh_texcoords: BindlessArray,
    pub mesh_instances: Buffer<MeshInstance>,
    pub accel: rtx::Accel,
}
pub struct Scene {
    pub textures: Polymorphic<PolyKey, dyn Texture>,
    pub surfaces: Polymorphic<PolyKey, dyn Surface>,
    pub lights: Polymorphic<PolyKey, dyn Light>,
    pub light_distribution: Box<dyn LightDistribution>,
    pub meshes: MeshAggregate,
}
impl MeshAggregate {
    pub fn triangle(&self, inst_id: Uint, prim_id: Uint) -> Expr<Triangle> {
        let inst = self.mesh_instances.var().read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.var().buffer::<Float3>(geom_id);
        let indices = self.mesh_indices.var().buffer::<Uint3>(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x());
        let v1 = vertices.read(i.y());
        let v2 = vertices.read(i.z());
        TriangleExpr::new(v0, v1, v2)
    }
    pub fn shading_triangle(&self, inst_id: Uint, prim_id: Uint) -> Expr<ShadingTriangle> {
        let inst = self.mesh_instances.var().read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.var().buffer::<Float3>(geom_id);
        let indices = self.mesh_indices.var().buffer::<Uint3>(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x());
        let v1 = vertices.read(i.y());
        let v2 = vertices.read(i.z());
        let (tc0, tc1, tc2) = if_!(inst.has_texcoord(), {
            let texcoords = self.mesh_texcoords.var().buffer::<Float2>(inst.texcoord_index());
            let indices = self.mesh_texcoords.var().buffer::<Uint3>(inst.texcoord_index());
            let i = indices.read(prim_id);
            let tc0 = texcoords.read(i.x());
            let tc1 = texcoords.read(i.y());
            let tc2 = texcoords.read(i.z());
            (tc0, tc1, tc2)
        }, else {
            let tc0 = make_float2(0.0, 0.0);
            let tc1 = make_float2(1.0, 0.0);
            let tc2 = make_float2(0.0, 0.1);
            (tc0, tc1, tc2)
        });
        let ng = (v2 - v0).cross(v1 - v0).normalize();
        let (n0, n1, n2) = if_!(inst.has_normal(), {
            let normals = self.mesh_normals.var().buffer::<Float3>(inst.normal_index());
            let indices = self.mesh_normals.var().buffer::<Uint3>(inst.normal_index());
            let i = indices.read(prim_id);
            let n0 = normals.read(i.x());
            let n1 = normals.read(i.y());
            let n2 = normals.read(i.z());
            (n0, n1, n2)
        }, else {
            (ng, ng, ng)
        });
        ShadingTriangleExpr::new(v0, v1, v2, tc0, tc1, tc2, n0, n1, n2, ng)
    }
}
impl Scene {
    pub fn intersect(&self, ray: Expr<Ray>) -> Expr<SurfaceInteraction> {
        let ro = ray.o();
        let rd = ray.d();
        let rtx_ray = RtxRayExpr::new(
            ro.x(),
            ro.y(),
            ro.z(),
            ray.t_min(),
            rd.x(),
            rd.y(),
            rd.z(),
            ray.t_max(),
        );
        let hit = self.meshes.accel.var().trace_closest(rtx_ray);
        if_!(hit.valid(), {
            let inst_id = hit.inst_id();
            let prim_id = hit.prim_id();
            let bary = make_float2(hit.u(), hit.v());
            let shading_triangle = self.meshes.shading_triangle(inst_id, prim_id);
            let p = shading_triangle.p(bary);
            let n = shading_triangle.n(bary);
            let uv = shading_triangle.tc(bary);
            let geometry = SurfaceLocalGeometryExpr::new(p, shading_triangle.ng(), n, uv, Float3Expr::zero(), Float3Expr::zero());
            SurfaceInteractionExpr::new(geometry, bary, prim_id, inst_id, FrameExpr::from_n(n), shading_triangle, Bool::from(true))
        }, else {
            zeroed::<SurfaceInteraction>().set_valid(Bool::from(false))
        })
    }
    pub fn occlude(&self, ray: Expr<Ray>) -> Expr<bool> {
        let ro = ray.o();
        let rd = ray.d();
        let rtx_ray = RtxRayExpr::new(
            ro.x(),
            ro.y(),
            ro.z(),
            ray.t_min(),
            rd.x(),
            rd.y(),
            rd.z(),
            ray.t_max(),
        );
        self.meshes.accel.var().trace_any(rtx_ray)
    }
}

struct SceneLoader {
    device: Device,
    parent_path: PathBuf,
    graph: Rc<node::Scene>,
    bsdfs: PolymorphicBuilder<PolyKey, dyn Bsdf>,
    texturse: PolymorphicBuilder<PolyKey, dyn Texture>,
}
