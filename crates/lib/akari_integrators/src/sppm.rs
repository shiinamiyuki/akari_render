use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicU32;

use akari_common::glam::vec3a;
use bumpalo::Bump;

use crate::bsdf::*;
// use crate::camera::*;
use crate::film::*;

// use crate::light::*;
use crate::sampler::*;
use crate::scene::*;
use crate::shape::*;
use crate::texture::ShadingPoint;
use crate::util::PerThread;
use crate::*;
#[derive(Clone)]
struct VisiblePoint<'a> {
    pub bsdf: BsdfClosure<'a>,
    pub p: Vec3A,
    pub wo: Vec3A,
    pub beta: SampledSpectrum,
    pub secondary_terminated: bool,
    // pub lambda: SampledWavelengths,
}

#[allow(non_snake_case)]
struct SppmPixel<'a> {
    radius: f32,
    ld: XYZ,
    M: AtomicU32,
    N: f32,
    phi: [AtomicFloat; 3],
    tau: XYZ,
    vp: Option<VisiblePoint<'a>>,
}

impl<'a> Clone for SppmPixel<'a> {
    fn clone(&self) -> Self {
        Self {
            radius: self.radius,
            ld: self.ld,
            M: AtomicU32::new(self.M.load(Ordering::Relaxed)),
            phi: self.phi.clone(),
            N: self.N,
            tau: self.tau,
            vp: self.vp.clone(),
        }
    }
}
struct SppmPixelListNode<'a> {
    pixel: &'a SppmPixel<'a>,
    next: AtomicPtr<SppmPixelListNode<'a>>,
}
// impl<'a> Drop for SppmPixelListNode<'a> {
//     fn drop(&mut self) {
//         unsafe {
//             let p = self.next.load(Ordering::Relaxed);
//             if !p.is_null() {
//                 Box::from_raw(p);
//             }
//         }
//     }
// }

struct SppmPixelList<'a>(AtomicPtr<SppmPixelListNode<'a>>);
impl<'a> Clone for SppmPixelList<'a> {
    fn clone(&self) -> Self {
        Self(AtomicPtr::new(self.0.load(Ordering::SeqCst)))
    }
}
struct VisiblePointGrid<'a> {
    bound: Bounds3f,
    grid: Vec<SppmPixelList<'a>>,
    hash_size: usize,
    grid_res: [u32; 3],
}
impl<'a> VisiblePointGrid<'a> {
    fn hash(&self, p: &UVec3) -> usize {
        ((p.x as usize * 73856093) ^ (p.y as usize * 19349663) ^ (p.z as usize * 83492791))
            % self.hash_size
    }
    pub fn new(bound: &Bounds3f, grid_res: [u32; 3], hash_size: usize) -> Self {
        Self {
            bound: *bound,
            grid: vec![SppmPixelList(AtomicPtr::new(std::ptr::null_mut())); hash_size],
            hash_size,
            grid_res,
        }
    }
    pub fn to_grid(&self, mut p: Vec3A) -> UVec3 {
        p = self.bound.max.min(p.into()).into();
        p = self.bound.min.max(p.into()).into();
        let mut q = self.bound.offset(p);
        q = q * vec3a(
            self.grid_res[0] as f32,
            self.grid_res[1] as f32,
            self.grid_res[2] as f32,
        );
        q.as_uvec3()
    }
    pub fn insert(&self, pixel: &'a SppmPixel<'a>) {
        if pixel.vp.is_none() {
            return;
        }
        let p = pixel.vp.as_ref().unwrap().p;
        let radius = pixel.radius;
        let pmin = self.to_grid(p - vec3a(radius, radius, radius));
        let pmax = self.to_grid(p + vec3a(radius, radius, radius));
        // println!("{:?} {:?}", pmin,pmax);
        for z in pmin.z..=pmax.z {
            for y in pmin.y..=pmax.y {
                for x in pmin.x..=pmax.x {
                    let h = self.hash(&uvec3(x, y, z));

                    self.insert_at(h, pixel);
                }
            }
        }
    }
    fn insert_at(&self, h: usize, pixel: &'a SppmPixel<'a>) {
        // let p = pixel.vp.as_ref().unwrap().p;
        // let h = self.hash(&self.to_grid(p));
        let ap = &self.grid[h].0;
        let node: *mut SppmPixelListNode<'a> = Box::into_raw(Box::new(SppmPixelListNode {
            pixel,
            next: AtomicPtr::new(ap.load(Ordering::SeqCst)),
        }));
        loop {
            unsafe {
                match ap.compare_exchange_weak(
                    (*node).next.load(Ordering::SeqCst),
                    node,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(x) => (*node).next.store(x, Ordering::SeqCst),
                }
            }
        }
    }
}

impl<'a> Drop for VisiblePointGrid<'a> {
    fn drop(&mut self) {
        for p in &self.grid {
            let mut p = p.0.load(Ordering::Relaxed);
            while !p.is_null() {
                unsafe {
                    let q = p;
                    p = (*p).next.load(Ordering::Relaxed);
                    Box::from_raw(q);
                }
            }
        }
    }
}
pub struct Sppm {
    pub iterations: usize,
    pub max_depth: usize,
    pub initial_radius: f32,
    pub n_photons: usize,
}

impl Integrator for Sppm {
    #[allow(non_snake_case)]
    fn render(&self, scene: &Scene) -> Film {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        let mut pixels: Vec<SppmPixel> = vec![
            SppmPixel {
                radius: self.initial_radius,
                ld: XYZ::zero(),
                N: 0.0,
                M: AtomicU32::new(0),
                tau: XYZ::zero(),
                vp: None,
                phi: Default::default(),
            };
            npixels
        ];
        let mut samplers: Vec<Box<dyn Sampler>> = vec![];
        for i in 0..npixels {
            samplers.push(Box::new(PCGSampler {
                rng: Pcg::new(i as u64),
            }));
        }
        let mut photon_samplers: Vec<Box<dyn Sampler>> = vec![];
        for i in 0..self.n_photons {
            photon_samplers.push(Box::new(PCGSampler {
                rng: Pcg::new(i as u64),
            }));
        }
        #[allow(unused_assignments)]
        let mut grid: Option<VisiblePointGrid> = None;

        let p_samplers = &UnsafePointer::new(samplers.as_mut_ptr());
        let p_pixels = &UnsafePointer::new(pixels.as_mut_ptr());
        let p_photon_samplers = &UnsafePointer::new(photon_samplers.as_mut_ptr());
        let mut arenas = PerThread::new(|| Bump::new());
        let progress = crate::util::create_progess_bar(self.iterations, "passes");
        for iter in 0..self.iterations {
            let pass_lambda = SampledWavelengths::sample_visible(radical_inverse(1, iter as u64));
            parallel_for(npixels, 256, |id| {
                let sppm_pixel = unsafe { p_pixels.offset(id as isize).as_mut().unwrap() };
                sppm_pixel.vp = None;
                sppm_pixel.M.store(0, Ordering::Relaxed);

                let x = (id as u32) % scene.camera.resolution().x;
                let y = (id as u32) / scene.camera.resolution().x;
                let pixel = uvec2(x, y);

                let sampler = unsafe { p_samplers.offset(id as isize).as_mut().unwrap().as_mut() };
                sampler.start_next_sample();
                let mut lambda = pass_lambda.clone();
                let (mut ray, _ray_weight) = scene.camera.generate_ray(pixel, sampler, &lambda);
                let mut beta = SampledSpectrum::one();
                let arena = arenas.get_mut();
                let mut depth = 0;
                loop {
                    if let Some(si) = scene.intersect(&ray) {
                        let opt_bsdf =
                            si.evaluate_bsdf(&mut lambda, TransportMode::CameraToLight, arena);
                        if opt_bsdf.is_none() {
                            return;
                        }
                        let p = ray.at(si.t);
                        let bsdf = opt_bsdf.unwrap();
                        let wo = -ray.d;
                        let sample = bsdf.sample(sampler.next2d(), wo);
                        if sample.is_none() {
                            break;
                        }
                        if depth >= self.max_depth {
                            break;
                        }
                        depth += 1;
                        let sample = sample.unwrap();
                        if sample.flag.contains(BsdfFlags::DIFFUSE)
                            || (sample.flag.contains(BsdfFlags::GLOSSY)
                                && depth == self.max_depth - 1)
                        {
                            sppm_pixel.vp = Some(VisiblePoint {
                                p,
                                beta,
                                bsdf,
                                wo,
                                secondary_terminated: lambda.secondary_terminated(),
                            });
                            break;
                        }
                        let wi = sample.wi;
                        ray = Ray::spawn(p, wi).offset_along_normal(si.ng);
                        beta *= sample.f * wi.dot(si.ng).abs() / sample.pdf;
                    } else {
                        break;
                    }
                }
            });
            {
                let mut bound = Bounds3f::default();
                let mut max_radius = 0.0 as f64;
                for pixel in &pixels {
                    if let Some(vp) = &pixel.vp {
                        let p_bound = Bounds3f {
                            min: (vp.p - vec3a(pixel.radius, pixel.radius, pixel.radius)).into(),
                            max: (vp.p + vec3a(pixel.radius, pixel.radius, pixel.radius)).into(),
                        };
                        bound.insert_box(p_bound);
                        max_radius = max_radius.max(pixel.radius as f64);
                    }
                }
                // log::info!("{:?} {}", bound, max_radius);
                let diag = bound.diagonal();
                let max_diag = diag.max_element() as f64;
                let base_grid_res = (max_diag / max_radius) as u32;
                let mut grid_res = [0; 3];
                for i in 0..3 {
                    grid_res[i] =
                        ((base_grid_res as f64 * diag[i] as f64 / max_diag) as u32).max(1);
                }
                grid = Some(VisiblePointGrid::new(&bound, grid_res, npixels));
                parallel_for(npixels, 256, |id| {
                    let sppm_pixel = unsafe { p_pixels.offset(id as isize).as_ref().unwrap() };
                    // let p = sppm_pixel.vp.as_ref().unwrap().p;
                    grid.as_ref().unwrap().insert(sppm_pixel);
                });
            }
            parallel_for(self.n_photons, 256, |id| {
                let sampler = unsafe {
                    p_photon_samplers
                        .offset(id as isize)
                        .as_mut()
                        .unwrap()
                        .as_mut()
                };
                sampler.start_next_sample();
                let mut lambda = pass_lambda.clone();
                let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
                let sample = light.sample_emission(sampler.next3d(), sampler.next2d(), &lambda);
                let mut depth = 0;
                let mut ray = sample.ray;
                let mut beta = sample.le / (sample.pdf_dir * sample.pdf_pos * light_pdf)
                    * sample.n.dot(ray.d).abs();
                let arena = arenas.get_mut();

                loop {
                    if let Some(si) = scene.intersect(&ray) {
                        let ng = si.ng;
                        let p = ray.at(si.t);
                        let opt_bsdf =
                            si.evaluate_bsdf(&mut lambda, TransportMode::LightToCamera, arena);
                        if opt_bsdf.is_none() {
                            break;
                        }
                        let bsdf = opt_bsdf.unwrap();
                        let wo = -ray.d;
                        // println!("{} {} {}", p, depth, self.max_depth);
                        depth += 1;
                        if depth >= self.max_depth {
                            break;
                        }

                        {
                            // splat to grid
                            let grid = grid.as_ref().unwrap();
                            let h = grid.hash(&grid.to_grid(p));
                            let mut ap = grid.grid[h].0.load(Ordering::Relaxed);
                            while !ap.is_null() {
                                let node = unsafe { &*ap };
                                let pixel = node.pixel;
                                let wi = -ray.d;
                                let vp = pixel.vp.as_ref().unwrap();
                                let dist2 = {
                                    let v = vp.p - p;
                                    v.length_squared()
                                };
                                if dist2 <= pixel.radius * pixel.radius {
                                    if vp.secondary_terminated {
                                        lambda.terminate_secondary();
                                    }
                                    let phi = lambda
                                        .cie_xyz(vp.beta * beta * vp.bsdf.evaluate(vp.wo, wi));
                                    for i in 0..3 {
                                        pixel.phi[i].fetch_add(phi[i] as f32, Ordering::SeqCst);
                                    }
                                    pixel.M.fetch_add(1, Ordering::Relaxed);
                                }
                                ap = node.next.load(Ordering::Relaxed);
                            }
                        }

                        if let Some(bsdf_sample) = bsdf.sample(sampler.next2d(), wo) {
                            let wi = bsdf_sample.wi;
                            ray = Ray::spawn(p, wi).offset_along_normal(ng);
                            beta *= bsdf_sample.f * wi.dot(ng).abs() / bsdf_sample.pdf;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            });
            parallel_for(npixels, 256, |id| {
                let p = unsafe { p_pixels.offset(id as isize).as_mut().unwrap() };
                let gamma = 2.0 / 3.0;

                if p.M.load(Ordering::Relaxed) > 0 {
                    let N_new = p.N + (gamma * p.M.load(Ordering::Relaxed) as f32);
                    let R_new =
                        p.radius * (N_new / (p.N + p.M.load(Ordering::Relaxed) as f32)).sqrt();
                    let mut phi = XYZ::zero();
                    for i in 0..3 {
                        phi[i] = p.phi[i].load(Ordering::Relaxed) as f32;
                        p.phi[i].store(0.0, Ordering::SeqCst);
                    }
                    p.tau = (p.tau + phi) * (R_new * R_new) / (p.radius * p.radius);
                    p.N = N_new;
                    p.radius = R_new;
                }
            });
            arenas.inner_mut().iter_mut().for_each(|a| a.reset());
            progress.inc(1);
        }
        parallel_for(npixels, 256, |id| {
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let p = unsafe { p_pixels.offset(id as isize).as_ref().unwrap() };
            let l = p.tau / ((self.iterations * self.n_photons) as f32 * PI * p.radius * p.radius);

            film.add_sample_xyz(pixel, l, 1.0);
        });
        film
    }
}
