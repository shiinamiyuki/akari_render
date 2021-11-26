use crate::bsdf::*;
use crate::camera::*;
use crate::film::*;
use crate::integrator::*;
use crate::light::*;
use crate::sampler::*;
use crate::scene::*;
use crate::shape::*;
use crate::*;
use crate::texture::ShadingPoint;
#[derive(Clone, Copy)]
pub struct VertexBase {
    pdf_fwd: Float,
    pdf_rev: Float,
    delta: bool,
    beta: Spectrum,
    wo: Vec3,
    p: Vec3,
    n: Vec3,
}
#[derive(Copy, Clone)]
pub struct SurfaceVertex<'a> {
    bsdf: BsdfClosure<'a>,
    n: Vec3,
    base: VertexBase,
}
#[derive(Copy, Clone)]
pub struct CameraVertex<'a> {
    camera: &'a dyn Camera,
    base: VertexBase,
}
#[derive(Copy, Clone)]
pub struct LightVertex<'a> {
    light: &'a dyn Light,
    base: VertexBase,
}
#[derive(Copy, Clone)]
pub enum Vertex<'a> {
    Camera(CameraVertex<'a>),
    Light(LightVertex<'a>),
    Surface(SurfaceVertex<'a>),
}

impl<'a> Vertex<'a> {
    fn create_camera_vertex(
        camera: &'a dyn Camera,
        ray: &Ray,
        beta: Spectrum,
        pdf_fwd: Float,
    ) -> Self {
        Self::Camera(CameraVertex {
            camera,
            base: VertexBase {
                wo: glm::zero(),
                pdf_fwd,
                pdf_rev: 0.0,
                delta: false,
                beta,
                p: ray.o,
                n: camera.n(), // ?????
            },
        })
    }
    fn create_light_vertex(light: &'a dyn Light, p: Vec3, beta: Spectrum, pdf_fwd: Float) -> Self {
        Self::Light(LightVertex {
            light,
            base: VertexBase {
                wo: glm::zero(),
                pdf_fwd,
                pdf_rev: 0.0,
                delta: false,
                beta,
                p,
                n: vec3(0.0, 0.0, 0.0), // ?????
            },
        })
    }
    fn create_surface_vertex(
        beta: Spectrum,
        p: Vec3,
        bsdf: BsdfClosure<'a>,
        wo: Vec3,
        n: Vec3,
        mut pdf_fwd: Float,
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
            Self::Light(v) => (v.light.flags() | LightFlags::DELTA) != LightFlags::NONE,
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
    pub fn pdf_light_origin(&self, scene: &Scene, next: &Vertex) -> Float {
        match self {
            Vertex::Light(v) => {
                let light_pdf = scene.light_distr.pdf(v.light);
                let wi = glm::normalize(&(next.p() - self.p()));
                let (pdf_pos, _) = v.light.pdf_li(
                    &wi,
                    &ReferencePoint {
                        p: self.p(),
                        n: self.n(),
                    },
                );
                light_pdf * pdf_pos
            }
            _ => unreachable!(),
        }
    }
    pub fn pdf_light(&self, _scene: &Scene, next: &Vertex) -> Float {
        match self {
            Vertex::Light(v) => {
                let ray = Ray::spawn_to(&self.p(), &next.p());
                let (_pdf_pos, pdf_dir) = v.light.pdf_le(&ray);
                self.convert_pdf_to_area(pdf_dir, next)
            }
            _ => unreachable!(),
        }
    }
    pub fn pdf(&self, scene: &Scene, prev: Option<&Vertex<'a>>, next: &Vertex<'a>) -> Float {
        let p2 = next.p();
        let p = self.p();
        let pdf = match self {
            Vertex::Surface(v) => {
                let p1 = prev.unwrap().p();
                let wo = glm::normalize(&(p1 - p));
                let wi = glm::normalize(&(p2 - p));
                v.bsdf.evaluate_pdf(&wo, &wi)
            }
            Vertex::Light(_) => self.pdf_light(scene, next),
            _ => unreachable!(),
        };
        self.convert_pdf_to_area(pdf, next)
    }
    pub fn f(&self, next: &Vertex<'a>, _mode: TransportMode) -> Spectrum {
        let v1 = self.as_surface().unwrap();
        // let v2 = next.as_surface().unwrap();
        let wi = glm::normalize(&(next.p() - self.p()));
        v1.bsdf.evaluate(&self.base().wo, &wi)
    }
    pub fn beta(&self) -> Spectrum {
        self.base().beta
    }
    pub fn pdf_fwd(&self) -> Float {
        self.base().pdf_fwd
    }
    pub fn p(&self) -> Vec3 {
        self.base().p
    }
    pub fn n(&self) -> Vec3 {
        self.base().n
    }
    pub fn le(&self, _scene: &'a Scene, prev: &Vertex<'a>) -> Spectrum {
        if let Some(v) = prev.as_light() {
            v.light.le(&Ray::spawn_to(&self.p(), &prev.p()))
        } else {
            Spectrum::zero()
        }
    }
    pub fn convert_pdf_to_area(&self, mut pdf: Float, v2: &Vertex) -> Float {
        let w = v2.p() - self.p();
        let inv_dist2 = 1.0 / glm::dot(&w, &w);
        if v2.on_surface() {
            pdf *= glm::dot(&v2.n(), &(w * inv_dist2.sqrt())).abs();
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
pub enum TransportMode {
    IMPORTANCE,
    RADIANCE,
}
pub fn random_walk<'a>(
    scene: &'a Scene,
    mut ray: Ray,
    sampler: &mut dyn Sampler,
    mut beta: Spectrum,
    pdf: Float,
    max_depth: usize,
    _mode: TransportMode,
    path: &mut Path<'a>,
) {
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
        if let Some(isct) = scene.shape.intersect(&ray) {
            let ng = isct.ng;
            let frame = Frame::from_normal(&ng);
            let shape = isct.shape.unwrap();
            let opt_bsdf = shape.bsdf();
            if opt_bsdf.is_none() {
                break;
            }
            let p = ray.at(isct.t);
            let bsdf = BsdfClosure {
                sp:ShadingPoint::from_intersection(&isct),
                frame,
                bsdf: opt_bsdf.unwrap(),
            };
            let wo = -ray.d;
            if depth >= max_depth {
                break;
            }
            let prev_index = depth;
            // let this_index = prev_index + 1;
            let prev = &mut path[prev_index];
            let vertex =
                Vertex::create_surface_vertex(beta, p, bsdf.clone(), wo, ng, pdf_fwd, prev);
            // pdf_rev = vertex.pdf(scene, prev, next)
            depth += 1;

            if let Some(bsdf_sample) = bsdf.sample(&sampler.next2d(), &wo) {
                pdf_fwd = bsdf_sample.pdf;
                let wi = &bsdf_sample.wi;
                {
                    pdf_rev = bsdf.evaluate_pdf(&wi, &wo);
                    prev.base_mut().pdf_rev = vertex.convert_pdf_to_area(pdf_rev, prev);
                }
                ray = Ray::spawn(&p, wi).offset_along_normal(&ng);
                beta *= bsdf_sample.f * glm::dot(wi, &ng).abs() / bsdf_sample.pdf;
                path.push(vertex);
            } else {
                break;
            }
        } else {
            break;
        }
    }
}
pub fn generate_camera_path<'a>(
    scene: &'a Scene,
    pixel: &glm::UVec2,
    sampler: &mut dyn Sampler,
    max_depth: usize,
    path: &mut Path<'a>,
) {
    path.clear();
    let camera = scene.camera.as_ref();
    let (ray, beta) = camera.generate_ray(pixel, sampler);
    let vertex = Vertex::create_camera_vertex(camera, &ray, beta, 1.0);
    path.push(vertex);
    let (_pdf_pos, pdf_dir) = camera.pdf_we(&ray);
    if pdf_dir == 0.0 {
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
    );
}
pub fn generate_light_path<'a>(
    scene: &'a Scene,
    sampler: &mut dyn Sampler,
    max_depth: usize,
    path: &mut Path<'a>,
) {
    path.clear();
    let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
    let sample = light.sample_le(&[sampler.next2d(), sampler.next2d()]);
    let le = sample.le;
    let beta = le / (sample.pdf_dir * sample.pdf_pos * light_pdf)
        * glm::dot(&sample.ray.d, &sample.n).abs();
    let vertex = Vertex::create_light_vertex(light, sample.ray.o, le, light_pdf * sample.pdf_pos);
    path.push(vertex);
    if sample.pdf_dir == 0.0{
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
    );
}
pub type Path<'a> = Vec<Vertex<'a>>;
pub fn geometry_term(scene: &Scene, v1: &Vertex, v2: &Vertex) -> Float {
    let mut wi = v1.p() - v2.p();
    let dist2: Float = glm::dot(&wi, &wi);
    wi /= dist2.sqrt();
    let ray = Ray::spawn_to(&v1.p(), &v2.p()).offset_along_normal(&v1.n());
    if scene.shape.occlude(&ray) {
        0.0
    } else {
        (glm::dot(&wi, &v1.n()) * glm::dot(&wi, &v2.n()) / dist2).abs()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ConnectionStrategy {
    pub s: usize,
    pub t: usize,
}
pub struct Scratch<'a> {
    new_light_path: Path<'a>,
    new_eye_path: Path<'a>,
    // strat:Option<ConnectionStrategy>,
}
impl<'a> Scratch<'a> {
    pub fn new() -> Self {
        Self {
            new_light_path: Vec::new(),
            new_eye_path: Vec::new(),
        }
    }
}
pub fn mis_weight<'a>(
    scene: &'a Scene,
    strat: ConnectionStrategy,
    original_light_path: &Path<'a>,
    original_eye_path: &Path<'a>,
    sampled: Option<Vertex<'a>>,
    scratch: &mut Scratch<'a>,
) -> Float {
    let s = strat.s;
    let t = strat.t;
    // 1.0 / (s + t - 1) as Float
    if s + t == 2 {
        return 1.0;
    }
    let eye_path = &mut scratch.new_eye_path;
    let light_path = &mut scratch.new_light_path;
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
        light_path.push(sampled.unwrap());
    } else if t == 1 {
        eye_path.push(sampled.unwrap());
    }
    // update vertices
    {
        let qs = if s > 0 {
            &mut light_path[s - 1] as *mut Vertex<'a>
        } else {
            std::ptr::null_mut()
        };
        let qs_minus = if s > 1 {
            &mut light_path[s - 2] as *mut Vertex<'a>
        } else {
            std::ptr::null_mut()
        };
        let pt = if t > 0 {
            &mut eye_path[t - 1] as *mut Vertex<'a>
        } else {
            std::ptr::null_mut()
        };
        let pt_minus = if t > 1 {
            &mut eye_path[t - 2] as *mut Vertex<'a>
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
            let pt = unsafe { pt.as_mut().unwrap() };
            let qs = unsafe { qs.as_ref() };
            let pt_minus = unsafe { pt_minus.as_ref().unwrap() };
            // pt-1 <- pt qs qs-1
            pt.base_mut().pdf_rev = if s > 0 {
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
    {
        // camera path
        let mut ri = 1.0;
        for i in (2..=t - 1).rev() {
            ri *= remap(eye_path[i].base().pdf_rev) / remap(eye_path[i].base().pdf_fwd);
            if !eye_path[i].base().delta {
                sum_ri += ri;
            }
        }
    }
    {
        let mut ri = 1.0;
        for i in (0..=s - 1).rev() {
            ri *= remap(light_path[i].base().pdf_rev) / remap(light_path[i].base().pdf_fwd);
            let delta_light = if i > 0 {
                light_path[i - 1].base().delta
            } else {
                light_path[0].is_delta_light()
            };
            if !light_path[i].base().delta && !delta_light {
                sum_ri += ri;
            }
        }
    }
    // println!("{}", 1.0 / (1.0 + sum_ri));
    1.0 / (1.0 + sum_ri)
}
pub fn connect_paths<'a>(
    scene: &'a Scene,
    strat: ConnectionStrategy,
    light_path: &Path<'a>,
    eye_path: &Path<'a>,
    sampler: &mut dyn Sampler,
    scratch: &mut Scratch<'a>,
) -> Spectrum {
    let s = strat.s;
    let t = strat.t;
    let mut sampled: Option<Vertex> = None;
    let mut l = Spectrum::zero();
    if t > 1 && s != 0 && eye_path[t - 1].as_light().is_some() {
        return Spectrum::zero();
    } else if s == 0 {
        let pt = &eye_path[t - 1];
        l = pt.beta() * pt.le(scene, &eye_path[t - 2]);
    } else if t == 1 {
        unreachable!();
    } else if s == 1 {
        let pt = &eye_path[t - 1];
        if pt.connectible() {
            let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
            let p_ref = ReferencePoint {
                p: pt.p(),
                n: pt.n(),
            };
            let light_sample = light.sample_li(&sampler.next3d(), &p_ref);
            if !light_sample.li.is_black() {
                {
                    let mut v = Vertex::create_light_vertex(
                        light,
                        light_sample.p,
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
                    l *= glm::dot(&light_sample.wi, &p_ref.n).abs();
                }
                if scene.shape.occlude(&light_sample.shadow_ray) {
                    l *= 0.0;
                }
                // li += beta
                //     * bsdf.evaluate(&wo, &light_sample.wi)
                //     * glm::dot(&ng, &light_sample.wi).abs()
                //     * light_sample.li
                //     / light_sample.pdf;
            }
        }
    } else {
        let pt = &eye_path[t - 1];
        let qs = &light_path[s - 1];
        if pt.connectible() && qs.connectible() {
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
        bdpt::mis_weight(scene, strat, light_path, eye_path, sampled, scratch)
    };
    l * mis_weight
}

pub struct Bdpt {
    pub spp: u32,
    pub max_depth: usize,
    pub debug: bool,
}

impl Integrator for Bdpt {
    fn render(&mut self, scene: &Scene) -> Film {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        let mut pyramid = Vec::new();
        for _t in 2..=self.max_depth + 2 {
            for _s in 0..self.max_depth + 2 {
                pyramid.push(Film::new(&scene.camera.resolution()));
            }
        }
        let get_index = |s, t| (t - 2) as usize * (3 + self.max_depth) + s as usize;
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            let mut camera_path = vec![];
            let mut light_path = vec![];
            let mut debug_acc = vec![];
            if self.debug {
                for _t in 2..=self.max_depth + 2 {
                    for _s in 0..=self.max_depth + 2 {
                        debug_acc.push(Spectrum::zero());
                    }
                }
            }
            let mut scratch = bdpt::Scratch::new();
            for _ in 0..self.spp {
                bdpt::generate_camera_path(
                    scene,
                    &pixel,
                    &mut sampler,
                    self.max_depth + 2,
                    &mut camera_path,
                );
                bdpt::generate_light_path(scene, &mut sampler, self.max_depth + 1, &mut light_path);
                for t in 2..=camera_path.len() as isize {
                    for s in 0..=light_path.len() as isize {
                        let depth = s + t - 2;
                        if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                            continue;
                        }
                        let li = bdpt::connect_paths(
                            scene,
                            bdpt::ConnectionStrategy {
                                s: s as usize,
                                t: t as usize,
                            },
                            &mut light_path,
                            &mut camera_path,
                            &mut sampler,
                            &mut scratch,
                        );
                        if self.debug {
                            debug_acc[get_index(s, t)] += li;
                        }
                        acc_li += li;
                    }
                }
            }
            acc_li = acc_li / (self.spp as Float);

            film.add_sample(&uvec2(x, y), &acc_li, 1.0);

            if self.debug {
                for t in 2..=(self.max_depth + 2) as isize {
                    for s in 0..=(self.max_depth + 2) as isize {
                        let depth = s + t - 2;
                        if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                            continue;
                        }
                        let idx = get_index(s, t);
                        pyramid[idx].add_sample(
                            &uvec2(x, y),
                            &(debug_acc[idx] / (self.spp as Float) as Float),
                            1.0,
                        );
                    }
                }
            }
        });
        if self.debug {
            for t in 2..=(self.max_depth + 2) as isize {
                for s in 0..=(self.max_depth + 2) as isize {
                    let depth = s + t - 2;
                    if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                        continue;
                    }
                    let idx = get_index(s, t);
                    let film = &pyramid[idx];
                    let img = film.to_rgb_image();
                    img.save(format!("bdpt-d{}-s{}-t{}.png", depth, s, t))
                        .unwrap();
                }
            }
        }
        film
    }
}
