use std::ops::DerefMut;

use bumpalo::Bump;

use crate::bsdf::*;
use crate::camera::*;
use crate::light::*;
use crate::sampler::*;
use crate::scene::*;
use crate::shape::*;
use crate::texture::ShadingPoint;
use crate::*;
#[derive(Clone, Copy)]
pub struct VertexBase {
    pub pdf_fwd: f32,
    pub pdf_rev: f32,
    pub delta: bool,
    pub beta: Spectrum,
    pub wo: Vec3,
    pub p: Vec3,
    pub n: Vec3,
}
#[derive(Copy, Clone)]
pub struct SurfaceVertex<'a> {
    pub bsdf: BsdfClosure<'a>,
    pub n: Vec3,
    pub base: VertexBase,
}
#[derive(Copy, Clone)]
pub struct CameraVertex<'a> {
    pub camera: &'a dyn Camera,
    pub base: VertexBase,
}
#[derive(Copy, Clone)]
pub struct LightVertex<'a> {
    pub light: &'a dyn Light,
    pub base: VertexBase,
}
#[derive(Copy, Clone)]
pub enum Vertex<'a> {
    Camera(CameraVertex<'a>),
    Light(LightVertex<'a>),
    Surface(SurfaceVertex<'a>),
}

pub struct Path<'a> {
    vertices: *mut Vertex<'a>,
    len: usize,
    capacity: usize,
}
impl<'a> std::ops::Deref for Path<'a> {
    type Target = [Vertex<'a>];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.vertices, self.len) }
    }
}
impl<'a> std::ops::DerefMut for Path<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.vertices, self.len) }
    }
}
impl<'a> Path<'a> {
    pub fn new<'b: 'a>(arena: &'b Bump, capacity: usize) -> Self {
        Self {
            capacity,
            len: 0,
            vertices: arena
                .alloc_layout(std::alloc::Layout::array::<Vertex<'a>>(capacity).unwrap())
                .as_ptr() as *mut Vertex<'a>,
        }
    }
    pub fn push(&mut self, v: Vertex<'a>) {
        if self.len == self.capacity {
            panic!("too many vertice");
        } else {
            unsafe { std::ptr::write(self.vertices.offset(self.len as isize), v) }
            self.len += 1;
        }
    }
    pub fn clear(&mut self) {
        for i in 0..self.len {
            unsafe {
                std::mem::drop(std::ptr::read(self.vertices.offset(i as isize)));
            }
        }
        self.len = 0;
    }
}
impl<'a> Vertex<'a> {
    pub fn create_camera_vertex(
        camera: &'a dyn Camera,
        ray: &Ray,
        beta: Spectrum,
        pdf_fwd: f32,
    ) -> Self {
        Self::Camera(CameraVertex {
            camera,
            base: VertexBase {
                wo: Vec3::ZERO,
                pdf_fwd,
                pdf_rev: 0.0,
                delta: false,
                beta,
                p: ray.o,
                n: camera.n(), // ?????
            },
        })
    }
    pub fn create_light_vertex(
        light: &'a dyn Light,
        p: Vec3,
        n: Vec3,
        beta: Spectrum,
        pdf_fwd: f32,
    ) -> Self {
        Self::Light(LightVertex {
            light,
            base: VertexBase {
                wo: Vec3::ZERO,
                pdf_fwd,
                pdf_rev: 0.0,
                delta: false,
                beta,
                p,
                n,
            },
        })
    }
    pub fn create_surface_vertex(
        beta: Spectrum,
        p: Vec3,
        bsdf: BsdfClosure<'a>,
        wo: Vec3,
        n: Vec3,
        mut pdf_fwd: f32,
        prev: &Vertex<'a>,
    ) -> Self {
        let mut v = Self::Surface(SurfaceVertex {
            bsdf,
            n,
            base: VertexBase {
                beta,
                wo,
                pdf_fwd: 0.0,
                pdf_rev: 0.0,
                delta: false,
                p,
                n,
            },
        });
        pdf_fwd = prev.convert_pdf_to_area(pdf_fwd, &v);
        v.base_mut().pdf_fwd = pdf_fwd;
        v
    }
    pub fn is_delta_light(&self) -> bool {
        match self {
            Self::Light(v) => v.light.flags().intersects( LightFlags::DELTA),//(v.light.flags() | LightFlags::DELTA) != LightFlags::NONE,
            _ => unreachable!(),
        }
    }
    pub fn on_surface(&self) -> bool {
        match self {
            Self::Camera(_v) => false,
            Self::Surface(_v) => true,
            Self::Light(_v) => false, //????
        }
    }
    pub fn as_camera(&self) -> Option<&CameraVertex> {
        match self {
            Self::Camera(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_surface(&self) -> Option<&SurfaceVertex> {
        match self {
            Self::Surface(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_light(&self) -> Option<&LightVertex> {
        match self {
            Self::Light(v) => Some(v),
            _ => None,
        }
    }
    pub fn base(&self) -> &VertexBase {
        match self {
            Self::Surface(v) => &v.base,
            Self::Light(v) => &v.base,
            Self::Camera(v) => &v.base,
        }
    }
    pub fn base_mut(&mut self) -> &mut VertexBase {
        match self {
            Self::Surface(v) => &mut v.base,
            Self::Light(v) => &mut v.base,
            Self::Camera(v) => &mut v.base,
        }
    }
    pub fn pdf_light_origin(&self, scene: &Scene, next: &Vertex) -> f32 {
        match self {
            Vertex::Light(v) => {
                let light_pdf = scene.light_distr.pdf(v.light);
                let (pdf_pos, _) = v.light.pdf_le(&Ray::spawn_to(self.p(), next.p()), self.n());
                light_pdf * pdf_pos
            }
            _ => unreachable!(),
        }
    }
    pub fn pdf_light(&self, _scene: &Scene, next: &Vertex) -> f32 {
        match self {
            Vertex::Light(v) => {
                let ray = Ray::spawn_to(self.p(), next.p());
                let (_pdf_pos, pdf_dir) = v.light.pdf_le(&ray, self.n());
                self.convert_pdf_to_area(pdf_dir, next)
            }
            _ => unreachable!(),
        }
    }
    pub fn pdf(&self, scene: &Scene, prev: Option<&Vertex<'a>>, next: &Vertex<'a>) -> f32 {
        let p2 = next.p();
        let p = self.p();
        match self {
            Vertex::Surface(v) => {
                let p1 = prev.unwrap().p();
                let wo = (p1 - p).normalize();
                let wi = (p2 - p).normalize();
                self.convert_pdf_to_area(v.bsdf.evaluate_pdf(wo, wi), next)
            }
            Vertex::Light(_) => self.pdf_light(scene, next),
            _ => unreachable!(),
        }
    }
    pub fn f(&self, next: &Vertex<'a>, _mode: TransportMode) -> Spectrum {
        let v1 = self.as_surface().unwrap();
        // let v2 = next.as_surface().unwrap();
        let wi = (next.p() - self.p()).normalize();
        v1.bsdf.evaluate(self.base().wo, wi)
    }
    pub fn beta(&self) -> Spectrum {
        self.base().beta
    }
    pub fn pdf_fwd(&self) -> f32 {
        self.base().pdf_fwd
    }
    pub fn p(&self) -> Vec3 {
        self.base().p
    }
    pub fn n(&self) -> Vec3 {
        self.base().n
    }
    pub fn le(&self, _scene: &'a Scene, prev: &Vertex<'a>) -> Spectrum {
        if let Some(v) = self.as_light() {
            let mut ray = Ray::spawn_to(prev.p(), self.p());
            ray.tmax *= 1.0 + 1e-3;
            v.light.le(&ray)
        } else {
            Spectrum::zero()
        }
    }
    pub fn convert_pdf_to_area(&self, mut pdf: f32, v2: &Vertex) -> f32 {
        let w = v2.p() - self.p();
        let inv_dist2 = 1.0 / w.length_squared();
        if v2.on_surface() {
            pdf *= v2.n().dot(w * inv_dist2.sqrt()).abs();
        }
        pdf * inv_dist2
    }
    pub fn connectible(&self) -> bool {
        match self {
            Vertex::Light(v) => {
                let flags = v.light.flags();
                (flags | LightFlags::DELTA_DIRECTION) == LightFlags::NONE
            }
            Vertex::Camera(_) => true,
            Vertex::Surface(_v) => true,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMode {
    IMPORTANCE,
    RADIANCE,
}
pub fn random_walk<'a, 'b>(
    scene: &'a Scene,
    mut ray: Ray,
    sampler: &mut dyn Sampler,
    mut beta: Spectrum,
    pdf: f32,
    max_depth: usize,
    mode: TransportMode,
    path: &mut Path<'b>,
    arena: &'b Bump,
) where
    'a: 'b,
{
    assert!(pdf > 0.0, "pdf is {}", pdf);
    assert!(path.len() == 1);
    if max_depth == 0 {
        return;
    }
    let mut pdf_fwd = pdf;
    #[allow(unused_assignments)]
    let mut pdf_rev = 0.0;
    let mut depth = 0usize;
    loop {
        if let Some(si) = scene.intersect(&ray) {
            let ng = si.ng;
            let shape = si.shape;
            let prev_index = depth;
            let prev = &mut path[prev_index];
            if mode == TransportMode::RADIANCE {
                if let Some(light) = scene.get_light_of_shape(shape) {
                    let mut vertex =
                        Vertex::create_light_vertex(light, ray.at(si.t), si.ng, beta, pdf_fwd);
                    vertex.base_mut().pdf_fwd = prev.convert_pdf_to_area(pdf_fwd, &vertex);
                    path.push(vertex);
                    break;
                }
            }
            let opt_bsdf = si.evaluate_bsdf(arena);
            if opt_bsdf.is_none() {
                break;
            }
            let p = ray.at(si.t);
            let bsdf = opt_bsdf.unwrap();
            let wo = -ray.d;

            let vertex =
                Vertex::create_surface_vertex(beta, p, bsdf.clone(), wo, ng, pdf_fwd, prev);
            // pdf_rev = vertex.pdf(scene, prev, next)
            std::mem::drop(prev);
            path.push(vertex);
            depth += 1;
            if depth >= max_depth {
                break;
            }
            if let Some(bsdf_sample) = bsdf.sample(sampler.next2d(), wo) {
                pdf_fwd = bsdf_sample.pdf;
                let wi = bsdf_sample.wi;
                let prev = &mut path[prev_index];
                {
                    pdf_rev = bsdf.evaluate_pdf(wi, wo);
                    prev.base_mut().pdf_rev = vertex.convert_pdf_to_area(pdf_rev, prev);
                }
                ray = Ray::spawn(p, wi).offset_along_normal(ng);
                beta *= bsdf_sample.f * wi.dot(ng).abs() / bsdf_sample.pdf;
            } else {
                break;
            }
        } else {
            break;
        }
    }
}
pub fn generate_camera_path<'a, 'b>(
    scene: &'a Scene,
    pixel: UVec2,
    sampler: &mut dyn Sampler,
    max_depth: usize,
    path: &mut Path<'b>,
    arena: &'b Bump,
) where
    'a: 'b,
{
    assert!(max_depth > 0);
    path.clear();
    let camera = scene.camera.as_ref();
    let (ray, beta) = camera.generate_ray(pixel, sampler);
    let vertex = Vertex::create_camera_vertex(camera, &ray, beta, 1.0);
    path.push(vertex);
    let (_pdf_pos, pdf_dir) = camera.pdf_we(&ray);
    if !(pdf_dir > 0.0) {
        return;
    }
    random_walk(
        scene,
        ray,
        sampler,
        beta,
        pdf_dir,
        max_depth - 1,
        TransportMode::RADIANCE,
        path,
        arena,
    );
}
pub fn generate_light_path<'a, 'b>(
    scene: &'a Scene,
    sampler: &mut dyn Sampler,
    max_depth: usize,
    path: &mut Path<'b>,
    arena: &'b Bump,
) where
    'a: 'b,
{
    if max_depth == 0 {
        return;
    }
    path.clear();
    let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
    let sample = light.sample_le([sampler.next2d(), sampler.next2d()]);
    let le = sample.le;
    let beta =
        le / (sample.pdf_dir * sample.pdf_pos * light_pdf) * sample.ray.d.dot(sample.n).abs();
    let vertex = Vertex::create_light_vertex(
        light,
        sample.ray.o,
        sample.n,
        le,
        light_pdf * sample.pdf_pos,
    );
    path.push(vertex);
    if !(sample.pdf_dir > 0.0) {
        return;
    }
    random_walk(
        scene,
        sample.ray,
        sampler,
        beta,
        sample.pdf_dir,
        max_depth - 1,
        TransportMode::IMPORTANCE,
        path,
        arena,
    );
}
pub fn geometry_term(scene: &Scene, v1: &Vertex, v2: &Vertex) -> f32 {
    let mut wi = v1.p() - v2.p();
    let dist2: f32 = wi.length_squared();
    wi /= dist2.sqrt();
    let mut ray = Ray::spawn_to(v1.p(), v2.p()).offset_along_normal(v1.n());
    ray.tmax *= 0.997;
    if scene.occlude(&ray) {
        0.0
    } else {
        (wi.dot(v1.n()) * wi.dot(v2.n()) / dist2).abs()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ConnectionStrategy {
    pub s: usize,
    pub t: usize,
}

pub fn mis_weight<'a, 'b>(
    scene: &'a Scene,
    strat: ConnectionStrategy,
    original_light_path: &Path<'b>,
    original_eye_path: &Path<'b>,
    sampled: Option<Vertex<'b>>,
    light_path: &mut Path<'b>,
    eye_path: &mut Path<'b>,
) -> f32
where
    'a: 'b,
{
    let s = strat.s;
    let t = strat.t;
    // 1.0 / (s + t - 1) as f32
    if s + t == 2 {
        return 1.0;
    }

    // return 1.0 / (s + t - 1) as f32;
    eye_path.clear();
    light_path.clear();
    for i in 0..s {
        light_path.push(original_light_path[i]);
    }
    for i in 0..t {
        eye_path.push(original_eye_path[i]);
        // println!(
        //     "{} {}",
        //     original_eye_path[i].base().pdf_fwd,
        //     original_eye_path[i].base().pdf_rev
        // );
    }
    if s == 1 {
        light_path[0] = sampled.unwrap();
    } else if t == 1 {
        eye_path[0] = sampled.unwrap();
    }
    // update vertices
    {
        let qs = if s > 0 {
            &mut light_path[s - 1] as *mut Vertex<'b>
        } else {
            std::ptr::null_mut()
        };
        let qs_minus = if s > 1 {
            &mut light_path[s - 2] as *mut Vertex<'b>
        } else {
            std::ptr::null_mut()
        };
        let pt = if t > 0 {
            &mut eye_path[t - 1] as *mut Vertex<'b>
        } else {
            std::ptr::null_mut()
        };
        let pt_minus = if t > 1 {
            &mut eye_path[t - 2] as *mut Vertex<'b>
        } else {
            std::ptr::null_mut()
        };
        // p0....pt-1 pt  qs qs-1 ...q0
        if !pt.is_null() {
            let pt = unsafe { pt.as_mut().unwrap() };
            let qs_minus = unsafe { qs_minus.as_ref() };
            let qs = unsafe { qs.as_ref() };
            let pt_minus = unsafe { &*pt_minus };
            pt.base_mut().delta = false;
            // pt-1 pt<- qs qs-1
            pt.base_mut().pdf_rev = if s > 0 {
                qs.unwrap().pdf(scene, qs_minus, pt)
            } else {
                pt.pdf_light_origin(scene, pt_minus)
            };
        }

        if !pt_minus.is_null() {
            let pt = unsafe { pt.as_ref().unwrap() };
            let qs = unsafe { qs.as_ref() };
            let pt_minus = unsafe { pt_minus.as_mut().unwrap() };
            // pt-1 <- pt qs qs-1
            pt_minus.base_mut().pdf_rev = if s > 0 {
                pt.pdf(scene, qs, pt_minus)
            } else {
                // pt-1 <- pt
                pt.pdf_light(scene, pt_minus)
            };
        }

        if !qs.is_null() {
            let qs = unsafe { qs.as_mut().unwrap() };
            let pt = unsafe { pt.as_mut().unwrap() };
            let pt_minus = unsafe { pt_minus.as_ref() };
            qs.base_mut().delta = false;
            // pt-1 pt-> qs qs-1
            qs.base_mut().pdf_rev = pt.pdf(scene, pt_minus, qs);
        }
        if !qs_minus.is_null() {
            let qs = unsafe { qs.as_mut().unwrap() };
            let pt = unsafe { pt.as_ref() };
            let qs_minus = unsafe { qs_minus.as_mut().unwrap() };
            qs_minus.base_mut().pdf_rev = qs.pdf(scene, pt, qs_minus);
        }
    }

    let mut sum_ri = 0.0;
    let remap = |x| {
        if x == 0.0 {
            1.0
        } else {
            x
        }
    };
    if t > 0 {
        // camera path
        let mut ri = 1.0;
        for i in (2..=t - 1).rev() {
            ri *= remap(eye_path[i].base().pdf_rev) / remap(eye_path[i].base().pdf_fwd);
            // cond_dbg!(s == 0, eye_path[i].base().pdf_rev);
            // cond_dbg!(s == 0, eye_path[i].base().pdf_fwd);
            if !eye_path[i].base().delta {
                sum_ri += ri;
            }
        }
    }
    if s > 0 {
        let mut ri = 1.0;
        for i in (0..=s - 1).rev() {
            ri *= remap(light_path[i].base().pdf_rev) / remap(light_path[i].base().pdf_fwd);
            // cond_dbg!(s == 1, light_path[i].base().pdf_rev);
            // cond_dbg!(s == 1, light_path[i].base().pdf_fwd);
            let delta_light = if i > 0 {
                light_path[i - 1].base().delta
            } else {
                light_path[0].is_delta_light()
            };
            // assert!(!delta_light);
            if !light_path[i].base().delta && !delta_light {
                sum_ri += ri;
            }
        }
    }
    // cond_dbg!(s==0, sum_ri);
    // println!("{}", 1.0 / (1.0 + sum_ri));
    (1.0 / (1.0 as f64 + sum_ri as f64)) as f32
}
pub fn connect_paths<'a, 'b>(
    scene: &'a Scene,
    strat: ConnectionStrategy,
    light_path: &Path<'b>,
    eye_path: &Path<'b>,
    sampler: &mut dyn Sampler,
    new_light_path: &mut Path<'b>,
    new_eye_path: &mut Path<'b>,
) -> (Spectrum, f32)
where
    'a: 'b,
{
    let s = strat.s;
    let t = strat.t;
    let mut sampled: Option<Vertex> = None;
    let mut l = Spectrum::zero();
    if t > 1 && s != 0 && eye_path[t - 1].as_light().is_some() {
        return (Spectrum::zero(), 0.0);
    } else if s == 0 {
        let pt = &eye_path[t - 1];
        l = pt.beta() * pt.le(scene, &eye_path[t - 2]);
    } else if t == 1 {
        unreachable!();
    } else if s == 1 {
        let pt = &eye_path[t - 1];
        if pt.connectible() && !pt.as_light().is_some() {
            let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
            let p_ref = ReferencePoint {
                p: pt.p(),
                n: pt.n(),
            };
            let light_sample = light.sample_li(sampler.next3d(), &p_ref);
            if !light_sample.li.is_black() {
                {
                    let mut v = Vertex::create_light_vertex(
                        light,
                        light_sample.p,
                        light_sample.n,
                        light_sample.li / (light_sample.pdf * light_pdf),
                        0.0,
                    );
                    v.base_mut().pdf_fwd = v.pdf_light_origin(scene, pt);
                    sampled = Some(v);
                }
                {
                    let sampled = sampled.as_ref().unwrap();
                    l = pt.beta() * pt.f(sampled, TransportMode::RADIANCE) * sampled.beta();
                }

                if pt.on_surface() {
                    l *= light_sample.wi.dot(p_ref.n).abs();
                }
                if !l.is_black() {
                    if scene.occlude(&light_sample.shadow_ray) {
                        l *= 0.0;
                    }
                }
                // li += beta
                //     * bsdf.evaluate(wo, light_sample.wi)
                //     * ng.dot(light_sample.wi).abs()
                //     * light_sample.li
                //     / light_sample.pdf;
            }
        }
    } else {
        let pt = &eye_path[t - 1];
        let qs = &light_path[s - 1];
        if pt.connectible() && qs.connectible() && !pt.as_light().is_some() {
            l = qs.beta()
                * pt.beta()
                * pt.f(qs, TransportMode::RADIANCE)
                * qs.f(pt, TransportMode::IMPORTANCE);
            if !l.is_black() {
                l *= geometry_term(scene, pt, qs);
            }
        }
    }
    let mis_weight = if l.is_black() {
        0.0
    } else {
        mis_weight(
            scene,
            strat,
            light_path,
            eye_path,
            sampled,
            new_light_path,
            new_eye_path,
        )
    };
    (l, mis_weight)
}
