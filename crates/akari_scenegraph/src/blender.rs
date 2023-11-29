use std::ffi::c_void;

use crate::*;
use akari_common::catch_signal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct ImportMeshArgs {
    pub vertex_ptr: u64,
    pub loop_tri_ptr: u64,
    pub uv_ptr: u64,
    pub mesh_ptr: u64,
    pub name: String,
    pub num_vertices: usize,
    pub num_triangles: usize,
    pub has_multi_materials: bool,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct MLoopTri([u32; 3]);
use cpp_ext::{
    blender_util::{
        get_mesh_material_indices, get_mesh_split_normals, get_mesh_tangents,
        get_mesh_triangle_indices,
    },
    ParallelForContext,
};

pub fn import_blender_mesh(scene: &mut Scene, args: ImportMeshArgs) -> NodeRef<Geometry> {
    let mesh_ptr = args.mesh_ptr;
    let loop_tri_ptr = args.loop_tri_ptr;
    let mut vertices = vec![[0.0f32; 3]; args.num_vertices];
    let mut indices = vec![[u32::MAX; 3]; args.num_triangles];
    let mut uvs = vec![];
    let mut normals;
    let mut tangents;
    let mut materials = vec![0u32];
    unsafe {
        if args.uv_ptr != 0 {
            uvs = vec![[0.0f32; 2]; args.num_triangles * 3];
            {
                std::ptr::copy(args.uv_ptr as *const [f32; 2], uvs.as_mut_ptr(), uvs.len());
                let degenerate = uvs.par_iter().all(|uv| uv[0] == 0.0 && uv[1] == 0.0);
                if degenerate {
                    uvs = vec![];
                }
            }
        }
        {
            std::ptr::copy(
                args.vertex_ptr as *const [f32; 3],
                vertices.as_mut_ptr(),
                vertices.len(),
            );
        }
        {
            let ctx = ParallelForContext::new();
            get_mesh_triangle_indices(
                &ctx,
                mesh_ptr as *const _,
                loop_tri_ptr as *const _,
                args.num_triangles,
                indices.as_mut_ptr() as *mut u32,
            );
        }
        {
            normals = vec![[0.0f32; 3]; args.num_triangles * 3];

            let ctx = ParallelForContext::new();
            if !get_mesh_split_normals(
                &ctx,
                mesh_ptr as *const _,
                loop_tri_ptr as *const _,
                args.num_triangles,
                normals.as_mut_ptr() as *mut f32,
            ) {
                normals = vec![];
            }
        }
        {
            tangents = vec![[0.0f32; 3]; args.num_triangles * 3];
            {
                let ctx = ParallelForContext::new();
                if !get_mesh_tangents(
                    &ctx,
                    mesh_ptr as *const _,
                    loop_tri_ptr as *const _,
                    args.num_triangles,
                    tangents.as_mut_ptr() as *mut f32,
                ) {
                    tangents = vec![];
                }
            }
        }
        if args.has_multi_materials {
            materials = vec![0u32; args.num_triangles];
            {
                let ctx = ParallelForContext::new();
                get_mesh_material_indices(
                    &ctx,
                    mesh_ptr as *const _,
                    loop_tri_ptr as *const _,
                    args.num_triangles,
                    materials.as_mut_ptr(),
                );
            }
        }
    }
    let vertices = Buffer::from_vec(vertices);
    let indices = Buffer::from_vec(indices);

    let name = &args.name;
    let vertices = scene.add_buffer(Some(format!("{name}.vert")), vertices);
    let indices = scene.add_buffer(Some(format!("{name}.ind")), indices);
    let uvs = if !uvs.is_empty() {
        let uvs = Buffer::from_vec(uvs);
        Some(scene.add_buffer(Some(format!("{name}.uv")), uvs))
    } else {
        None
    };
    let normals = if !normals.is_empty() {
        let normals = Buffer::from_vec(normals);
        Some(scene.add_buffer(Some(format!("{name}.norm")), normals))
    } else {
        None
    };
    let tangents = if !tangents.is_empty() {
        let tangents = Buffer::from_vec(tangents);
        Some(scene.add_buffer(Some(format!("{name}.tang")), tangents))
    } else {
        None
    };
    let materials = Buffer::from_vec(materials);
    let materials = scene.add_buffer(Some(format!("{name}.mat")), materials);
    let mesh = scene.add_mesh(
        Some(name.clone()),
        vertices,
        indices,
        normals,
        uvs,
        tangents,
        materials,
    );
    mesh
}
