use crate::texture::ShadingPoint;
use crate::util::profile::scope;
use crate::*;
use crate::{shape::SurfaceInteraction, AsAny};
use akari_common::glam::vec3a;
use embree_sys as sys;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::ffi::c_void;
use std::{any::Any, collections::HashMap, convert::TryInto, sync::Arc};
use sys::RTCIntersectContext;

use crate::{
    bsdf::Bsdf,
    distribution::Distribution1D,
    shape::{MeshInstanceProxy, Shape, SurfaceSample, TriangleMesh},
    Bounds3f, Ray, Vec3A,
};
struct Device(sys::RTCDevice);
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

lazy_static! {
    static ref DEVICE: Mutex<Device> = Mutex::new(Device(std::ptr::null_mut()));
}
fn init_device() {
    let mut device = DEVICE.lock();
    if device.0.is_null() {
        device.0 = unsafe { sys::rtcNewDevice(std::ptr::null()) }
    }
}
struct EmbreeMeshAccel {
    scene: sys::RTCScene,
    #[allow(dead_code)]
    mesh: Arc<TriangleMesh>,
}
unsafe impl Send for EmbreeMeshAccel {}
unsafe impl Sync for EmbreeMeshAccel {}
impl EmbreeMeshAccel {
    unsafe fn new(mesh: Arc<TriangleMesh>) -> Self {
        init_device();
        let device = DEVICE.lock();
        let device = device.0;
        let scene = sys::rtcNewScene(device);
        let geometry = sys::rtcNewGeometry(device, sys::RTCGeometryType_RTC_GEOMETRY_TYPE_TRIANGLE);
        assert!(std::mem::size_of::<Vec3>() == std::mem::size_of::<[f32; 3]>());
        assert!(std::mem::align_of::<Vec3>() == std::mem::align_of::<[f32; 3]>());
        assert!(std::mem::size_of::<UVec3>() == std::mem::size_of::<[u32; 3]>());
        assert!(std::mem::align_of::<UVec3>() == std::mem::align_of::<[u32; 3]>());
        sys::rtcSetSharedGeometryBuffer(
            geometry,
            sys::RTCBufferType_RTC_BUFFER_TYPE_VERTEX,
            0,
            sys::RTCFormat_RTC_FORMAT_FLOAT3,
            mesh.vertices.as_ptr() as *const c_void,
            0,
            (3 * std::mem::size_of::<f32>()).try_into().unwrap(),
            mesh.vertices.len().try_into().unwrap(),
        );
        sys::rtcSetSharedGeometryBuffer(
            geometry,
            sys::RTCBufferType_RTC_BUFFER_TYPE_INDEX,
            0,
            sys::RTCFormat_RTC_FORMAT_UINT3,
            mesh.indices.as_ptr() as *const c_void,
            0,
            (3 * std::mem::size_of::<u32>()).try_into().unwrap(),
            mesh.indices.len().try_into().unwrap(),
        );
        // let vb = sys::rtcSetNewGeometryBuffer(
        //     geometry,
        //     sys::RTCBufferType_RTC_BUFFER_TYPE_VERTEX,
        //     0,
        //     sys::RTCFormat_RTC_FORMAT_FLOAT3,
        //     (3 * std::mem::size_of::<f32>()).try_into().unwrap(),
        //     mesh.vertices.len().try_into().unwrap(),
        // );

        // let vb = std::slice::from_raw_parts_mut(vb as *mut [f32; 3], mesh.vertices.len());
        // vb.par_iter_mut()
        //     .enumerate()
        //     .for_each(|(i, v)| *v = mesh.vertices[i].into());
        // let ib = sys::rtcSetNewGeometryBuffer(
        //     geometry,
        //     sys::RTCBufferType_RTC_BUFFER_TYPE_INDEX,
        //     0,
        //     sys::RTCFormat_RTC_FORMAT_UINT3,
        //     (3 * std::mem::size_of::<u32>()).try_into().unwrap(),
        //     mesh.indices.len().try_into().unwrap(),
        // );
        // let ib = std::slice::from_raw_parts_mut(ib as *mut [u32; 3], mesh.indices.len());
        // ib.par_iter_mut()
        //     .enumerate()
        //     .for_each(|(i, v)| *v = mesh.indices[i].into());
        sys::rtcCommitGeometry(geometry);
        sys::rtcAttachGeometry(scene, geometry);
        sys::rtcReleaseGeometry(geometry);
        sys::rtcCommitScene(scene);
        Self { scene, mesh }
    }
}
#[allow(dead_code)]
pub struct EmbreeInstance {
    base: sys::RTCScene,
    instance_scene: sys::RTCScene,
    instance: sys::RTCGeometry,
    mesh: Arc<dyn Shape>,
    mesh_ref: &'static MeshInstanceProxy,
    area: f32,
    dist: Distribution1D,
}
unsafe impl Send for EmbreeInstance {}
unsafe impl Sync for EmbreeInstance {}
impl EmbreeInstance {
    unsafe fn new(base: sys::RTCScene, mesh: Arc<dyn Shape>) -> Self {
        init_device();
        let device = DEVICE.lock();
        let device = device.0;
        let geometry = sys::rtcNewGeometry(device, sys::RTCGeometryType_RTC_GEOMETRY_TYPE_INSTANCE);
        sys::rtcSetGeometryInstancedScene(geometry, base);
        sys::rtcCommitGeometry(geometry);
        let scene = sys::rtcNewScene(device);
        sys::rtcAttachGeometry(scene, geometry);
        sys::rtcCommitScene(scene);
        let mesh_ref = mesh
            .as_ref()
            .as_any()
            .downcast_ref::<MeshInstanceProxy>()
            .unwrap();
        sys::rtcRetainScene(base);
        Self {
            base,
            instance_scene: scene,
            mesh: mesh.clone(),
            mesh_ref: std::mem::transmute(mesh_ref),
            instance: geometry,
            area: mesh_ref.mesh.area(),
            dist: mesh_ref.mesh.area_distribution(),
        }
    }
}
impl Drop for EmbreeInstance {
    fn drop(&mut self) {
        unsafe {
            sys::rtcReleaseGeometry(self.instance);
            sys::rtcReleaseScene(self.instance_scene);
            sys::rtcReleaseScene(self.base);
        }
    }
}

impl Shape for EmbreeInstance {
    fn intersect(&self, ray: &Ray, _: Option<Vec3A>) -> Option<RayHit> {
        let _profiler = scope("EmbreeInstance::intersect");
        unsafe {
            let mut rayhit = sys::RTCRayHit {
                ray: to_rtc_ray(ray),
                hit: sys::RTCHit {
                    Ng_x: 0.0,
                    Ng_y: 0.0,
                    Ng_z: 0.0,
                    u: 0.0,
                    v: 0.0,
                    primID: u32::MAX,
                    geomID: u32::MAX,
                    instID: [u32::MAX],
                },
            };
            let mut ctx = RTCIntersectContext {
                flags: sys::RTCIntersectContextFlags_RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT,
                filter: None,
                instID: [u32::MAX],
            };
            sys::rtcIntersect1(
                self.instance_scene,
                &mut ctx as *mut _,
                &mut rayhit as *mut _,
            );
            if rayhit.hit.geomID != u32::MAX {
                let uv = vec2(rayhit.hit.u, rayhit.hit.v);
                let ng = vec3a(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z).normalize();
                Some(RayHit {
                    uv,
                    t: rayhit.ray.tfar,
                    ng,
                    prim_id: rayhit.hit.primID,
                    geom_id: u32::MAX,
                })
            } else {
                None
            }
        }
    }
    fn occlude(&self, ray: &Ray, _: Option<Vec3A>) -> bool {
        let _profiler = scope("EmbreeInstance::occlude");
        unsafe {
            let mut ray = to_rtc_ray(ray);
            let mut ctx = RTCIntersectContext {
                flags: sys::RTCIntersectContextFlags_RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT,
                filter: None,
                instID: [u32::MAX],
            };
            sys::rtcOccluded1(self.instance_scene, &mut ctx as *mut _, &mut ray as *mut _);
            ray.tfar < 0.0
        }
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        self.mesh_ref.bsdf()
    }
    fn aabb(&self) -> Bounds3f {
        todo!()
    }
    fn sample_surface(&self, u: Vec3A) -> SurfaceSample {
        self.mesh_ref.mesh.sample_surface(u, &self.dist)
    }
    fn area(&self) -> f32 {
        self.area
    }

    fn shading_triangle<'a>(&'a self, prim_id: u32) -> shape::ShadingTriangle<'a> {
        self.mesh_ref.shading_triangle(prim_id)
    }

    fn triangle(&self, prim_id: u32) -> shape::Triangle {
        self.mesh_ref.triangle(prim_id)
    }
}
pub struct EmbreeTopLevelAccel {
    scene: sys::RTCScene,
    instances: Vec<Arc<EmbreeInstance>>,
}
unsafe impl Send for EmbreeTopLevelAccel {}
unsafe impl Sync for EmbreeTopLevelAccel {}
impl Drop for EmbreeTopLevelAccel {
    fn drop(&mut self) {
        unsafe {
            sys::rtcReleaseScene(self.scene);
        }
    }
}

impl EmbreeTopLevelAccel {
    pub(crate) unsafe fn new(shapes: &Vec<Arc<dyn Shape>>) -> Self {
        init_device();
        let mut cache: HashMap<*const dyn Any, EmbreeMeshAccel> = HashMap::new();
        let shapes: Vec<_> = shapes
            .iter()
            .map(|shape_| {
                let shape = shape_.as_ref().as_any();
                if let Some(mesh) = shape.downcast_ref::<MeshInstanceProxy>() {
                    let base = mesh.mesh.clone();
                    if !cache.contains_key(&Arc::as_ptr(&(base.clone() as Arc<dyn Any>))) {
                        let accel = EmbreeMeshAccel::new(base.clone());
                        cache.insert(Arc::as_ptr(&(base.clone() as Arc<dyn Any>)), accel);
                    }
                    let accel = cache
                        .get(&Arc::as_ptr(&(base.clone() as Arc<dyn Any>)))
                        .unwrap()
                        .clone();
                    Arc::new(EmbreeInstance::new(accel.scene, shape_.clone()))
                } else {
                    unimplemented!()
                }
            })
            .collect();
        let device = DEVICE.lock();
        let device = device.0;
        let scene = sys::rtcNewScene(device);
        for (id, shape) in shapes.iter().enumerate() {
            sys::rtcAttachGeometryByID(scene, shape.instance, id as u32);
        }
        sys::rtcCommitScene(scene);
        Self {
            scene,
            instances: shapes,
        }
    }
}


impl accel::Accel for EmbreeTopLevelAccel {
    fn shapes(&self) -> Vec<Arc<dyn Shape>> {
        self.instances
            .iter()
            .map(|x| x.clone() as Arc<dyn Shape>)
            .collect()
    }
    fn hit_to_iteraction<'a>(&'a self, rayhit: RayHit) -> SurfaceInteraction<'a> {
        let instance = &self.instances[rayhit.geom_id as usize];
        let triangle = instance.shading_triangle(rayhit.prim_id);
        let uv = rayhit.uv;
        let ns = triangle.ns(uv);
        let texcoord = triangle.texcoord(uv);
        SurfaceInteraction {
            shape: instance.as_ref(),
            bsdf: triangle.bsdf,
            triangle,
            t: rayhit.t,
            uv,
            ng: rayhit.ng,
            ns,
            sp: ShadingPoint { texcoord },
            texcoord,
        }
    }
    fn intersect4(&self, rays: &[Ray; 4], mask: [bool; 4]) -> [Option<RayHit>; 4] {
        let _profiler = scope("EmbreeTopLevelAccel::intersect4");
        let mut rayhit4 = sys::RTCRayHit4 {
            ray: to_rtc_ray4(rays),
            hit: sys::RTCHit4 {
                Ng_x: [0.0; 4],
                Ng_y: [0.0; 4],
                Ng_z: [0.0; 4],
                u: [0.0; 4],
                v: [0.0; 4],
                primID: [u32::MAX; 4],
                geomID: [u32::MAX; 4],
                instID: [[u32::MAX; 4]],
            },
        };
        let mut ctx = RTCIntersectContext {
            flags: sys::RTCIntersectContextFlags_RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT,
            filter: None,
            instID: [u32::MAX],
        };
        unsafe {
            let mut valid = [-1; 4];
            for i in 0..4 {
                if !mask[i] {
                    valid[i] = 0;
                }
            }
            sys::rtcIntersect4(
                &mut valid as *mut _,
                self.scene,
                &mut ctx as *mut _,
                &mut rayhit4 as *mut _,
            );
            let mut hits = [None; 4];
            for i in 0..4 {
                hits[i] = if rayhit4.hit.geomID[i] != u32::MAX {
                    let ng = vec3a(
                        rayhit4.hit.Ng_x[i],
                        rayhit4.hit.Ng_y[i],
                        rayhit4.hit.Ng_z[i],
                    )
                    .normalize();
                    let uv = vec2(rayhit4.hit.u[i], rayhit4.hit.v[i]);
                    Some(RayHit {
                        uv,
                        t: rayhit4.ray.tfar[i],
                        ng,
                        prim_id: rayhit4.hit.primID[i],
                        geom_id: rayhit4.hit.instID[0][i],
                    })
                } else {
                    None
                }
            }
            hits
        }
    }
    fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        let _profiler = scope("EmbreeTopLevelAccel::intersect");
        unsafe {
            let mut rayhit = sys::RTCRayHit {
                ray: to_rtc_ray(ray),
                hit: sys::RTCHit {
                    Ng_x: 0.0,
                    Ng_y: 0.0,
                    Ng_z: 0.0,
                    u: 0.0,
                    v: 0.0,
                    primID: u32::MAX,
                    geomID: u32::MAX,
                    instID: [u32::MAX],
                },
            };
            let mut ctx = RTCIntersectContext {
                flags: sys::RTCIntersectContextFlags_RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT,
                filter: None,
                instID: [u32::MAX],
            };
            sys::rtcIntersect1(self.scene, &mut ctx as *mut _, &mut rayhit as *mut _);
            if rayhit.hit.geomID != u32::MAX {
                let ng = vec3a(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z).normalize();
                let uv = vec2(rayhit.hit.u, rayhit.hit.v);
                Some(RayHit {
                    uv,
                    t: rayhit.ray.tfar,
                    ng,
                    prim_id: rayhit.hit.primID,
                    geom_id: rayhit.hit.instID[0],
                })
            } else {
                None
            }
        }
    }
    fn occlude4(&self, rays: &[Ray; 4], mask: [bool; 4]) -> [bool; 4] {
        let _profiler = scope("EmbreeTopLevelAccel::occlude4");
        let mut ray4 = to_rtc_ray4(rays);
        let mut ctx = RTCIntersectContext {
            flags: sys::RTCIntersectContextFlags_RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT,
            filter: None,
            instID: [u32::MAX],
        };
        unsafe {
            let mut valid = [-1; 4];
            for i in 0..4 {
                if !mask[i] {
                    valid[i] = 0;
                }
            }
            sys::rtcOccluded4(
                &mut valid as *mut _,
                self.scene,
                &mut ctx as *mut _,
                &mut ray4 as *mut _,
            );
            let mut occluded = [false; 4];
            for i in 0..4 {
                occluded[i] = ray4.tfar[i] < 0.0
            }
            occluded
        }
    }
    fn occlude(&self, ray: &Ray) -> bool {
        let _profiler = scope("EmbreeTopLevelAccel::occlude");
        unsafe {
            let mut ray = to_rtc_ray(ray);
            let mut ctx = RTCIntersectContext {
                flags: sys::RTCIntersectContextFlags_RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT,
                filter: None,
                instID: [u32::MAX],
            };
            sys::rtcOccluded1(self.scene, &mut ctx as *mut _, &mut ray as *mut _);
            ray.tfar < 0.0
        }
    }
}

fn to_rtc_ray4(ray: &[Ray; 4]) -> sys::RTCRay4 {
    sys::RTCRay4 {
        org_x: [ray[0].o.x, ray[1].o.x, ray[2].o.x, ray[3].o.x],
        org_y: [ray[0].o.y, ray[1].o.y, ray[2].o.y, ray[3].o.y],
        org_z: [ray[0].o.z, ray[1].o.z, ray[2].o.z, ray[3].o.z],
        dir_x: [ray[0].d.x, ray[1].d.x, ray[2].d.x, ray[3].d.x],
        dir_y: [ray[0].d.y, ray[1].d.y, ray[2].d.y, ray[3].d.y],
        dir_z: [ray[0].d.z, ray[1].d.z, ray[2].d.z, ray[3].d.z],
        time: [0.0; 4],
        tnear: [ray[0].tmin, ray[1].tmin, ray[2].tmin, ray[3].tmin],
        tfar: [ray[0].tmax, ray[1].tmax, ray[2].tmax, ray[3].tmax],
        id: [0; 4],
        mask: [0; 4],
        flags: [0; 4],
    }
}
fn to_rtc_ray(ray: &Ray) -> sys::RTCRay {
    let rtc_ray = sys::RTCRay {
        org_x: ray.o.x,
        org_y: ray.o.y,
        org_z: ray.o.z,
        dir_x: ray.d.x,
        dir_y: ray.d.y,
        dir_z: ray.d.z,
        tnear: ray.tmin,
        tfar: ray.tmax,
        time: 0.0,
        mask: 0,
        id: 0,
        flags: 0,
    };
    rtc_ray
}
