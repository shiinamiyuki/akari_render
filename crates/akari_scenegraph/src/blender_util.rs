use crate::*;
use serde::{Deserialize, Serialize};
use std::ffi::c_char;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportMeshArgs {
    pub vertex_ptr: u64,
    pub loop_tri_ptr: u64,
    pub uv_ptr: u64,
    pub mesh_ptr: u64,
    pub name: String,
    pub num_vertices: usize,
    pub num_triangles: usize,
    pub has_normals: bool,
    pub has_tangents: bool,
    pub has_uvs: bool,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct MLoopTri([u32; 3]);
#[link(name = "akari_blender_cpp_ext")]
extern "C" {
    // extern "C" void get_mesh_triangle_indices(const Mesh *mesh, const MLoopTri *tri, size_t count, int *out)
    fn get_mesh_triangle_indices(mesh_ptr: u64, tri_ptr: u64, count: usize, out: *mut u32);

    // extern "C" void get_mesh_tangents(const Mesh *mesh,  const MLoopTri *tri, size_t count, float *out)
    fn get_mesh_tangents(mesh_ptr: u64, tri_ptr: u64, count: usize, out: *mut f32);

    // extern "C" void get_mesh_split_normals(const Mesh *mesh, const MLoopTri *tri, size_t count, float *out)
    fn get_mesh_split_normals(mesh_ptr: u64, tri_ptr: u64, count: usize, out: *mut f32);

}

pub fn import_blender_mesh(scene: &mut Scene, args: ImportMeshArgs) -> NodeRef<Geometry> {
    let mesh_ptr = args.mesh_ptr;
    let loop_tri_ptr = args.loop_tri_ptr;
    let mut vertices = vec![[0.0f32; 3]; args.num_vertices];
    let mut indices = vec![[u32::MAX; 3]; args.num_triangles];
    let mut uvs = vec![];
    let mut normals = vec![];
    let mut tangents = vec![];

    rayon::scope(|s| unsafe {
        if args.has_uvs {
            uvs = vec![[0.0f32; 2]; args.num_triangles * 3];
            s.spawn(|_| {
                std::ptr::copy(args.uv_ptr as *const [f32; 2], uvs.as_mut_ptr(), uvs.len());
            });
        }
        s.spawn(|_| {
            std::ptr::copy(
                args.vertex_ptr as *const [f32; 3],
                vertices.as_mut_ptr(),
                vertices.len(),
            );
        });
        s.spawn(|_| {
            get_mesh_triangle_indices(
                mesh_ptr,
                loop_tri_ptr,
                args.num_triangles,
                indices.as_mut_ptr() as *mut u32,
            );
        });
        if args.has_normals {
            normals = vec![[0.0f32; 3]; args.num_triangles * 3];
            s.spawn(|_| {
                get_mesh_split_normals(
                    mesh_ptr,
                    loop_tri_ptr,
                    args.num_triangles,
                    normals.as_mut_ptr() as *mut f32,
                );
            });
        }
        if args.has_tangents {
            tangents = vec![[0.0f32; 3]; args.num_triangles * 3];
            s.spawn(|_| {
                get_mesh_tangents(
                    mesh_ptr,
                    loop_tri_ptr,
                    args.num_triangles,
                    tangents.as_mut_ptr() as *mut f32,
                );
            });
        }
    });
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
    let mesh = scene.add_mesh(
        Some(name.clone()),
        vertices,
        indices,
        normals,
        uvs,
        tangents,
    );
    mesh
}
