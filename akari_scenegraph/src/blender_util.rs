use crate::*;
use serde::{Deserialize, Serialize};
use std::ffi::c_char;

#[derive(Serialize, Deserialize)]
struct ExportMeshArgs {
    vertex_ptr: u64,
    loop_tri_ptr: u64,
    uv_ptr: u64,
    mesh_ptr: u64,
    out_path: String,
    num_vertices: usize,
    num_triangles: usize,
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

#[no_mangle]
pub unsafe extern "cdecl" fn export_blender_mesh(json_args: *const c_char) {
    let args = unsafe {
        let cstr = std::ffi::CStr::from_ptr(json_args);
        let str_slice: &str = cstr.to_str().unwrap();
        let v: ExportMeshArgs = serde_json::from_str(str_slice).unwrap();
        v
    };
    let mesh_ptr = args.mesh_ptr;
    let out_path = &args.out_path;
    let loop_tri_ptr = args.loop_tri_ptr;
    let mut vertices = vec![[0.0f32; 3]; args.num_vertices];
    let mut indices = vec![[u32::MAX; 3]; args.num_triangles];
    let mut uvs = vec![[0.0f32; 2]; args.num_triangles * 3];
    let mut normals = vec![[0.0f32; 3]; args.num_triangles * 3];
    let mut tangents = vec![[0.0f32; 3]; args.num_triangles * 3];

    rayon::scope(|s| {
        s.spawn(|_| {
            std::ptr::copy(args.uv_ptr as *const [f32; 2], uvs.as_mut_ptr(), uvs.len());
        });
        s.spawn(|_| {
            std::ptr::copy(
                args.vertex_ptr as *const [f32; 3],
                vertices.as_mut_ptr(),
                vertices.len(),
            );
        });
        s.spawn(|_| {
            get_mesh_split_normals(
                mesh_ptr,
                loop_tri_ptr,
                args.num_triangles,
                normals.as_mut_ptr() as *mut f32,
            );
        });
        s.spawn(|_| {
            get_mesh_tangents(
                mesh_ptr,
                loop_tri_ptr,
                args.num_triangles,
                tangents.as_mut_ptr() as *mut f32,
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
    });
    write_binary(format!("{}.vert", out_path), &vertices).unwrap();
    write_binary(format!("{}.ind", out_path), &indices).unwrap();
    write_binary(format!("{}.uv", out_path), &uvs).unwrap();
    write_binary(format!("{}.normal", out_path), &normals).unwrap();
    write_binary(format!("{}.tangent", out_path), &tangents).unwrap();
}
