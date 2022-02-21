use std::collections::HashSet;
use std::iter::once;

use crate::bsdf::*;
// use crate::camera::*;
use crate::film::*;

use crate::light::*;
use crate::sampler::*;
use crate::scene::*;
use crate::util::nn_v2::*;
use crate::util::PerThread;
use crate::*;

use bumpalo::Bump;
use nalgebra as na;
use rand::Rng;

const POSITION_INPUTS: usize = 3;
const POSITION_FREQ: usize = 12;
const NOMRAL_INPUTS: usize = 2;
const DIR_INPUTS: usize = 2;
const ALBEDO_INPUTS: usize = 3;
const ROUGHNESS_INPUTS: usize = 1;
const METALLIC_INPUTS: usize = 1;
const ONE_BLOB_SIZE: usize = 4;
const INPUT_SIZE: usize = POSITION_INPUTS
    + NOMRAL_INPUTS
    + DIR_INPUTS
    + ALBEDO_INPUTS
    + METALLIC_INPUTS
    + ROUGHNESS_INPUTS;
const MAPPED_SIZE: usize = POSITION_INPUTS * POSITION_FREQ
    + ONE_BLOB_SIZE * (NOMRAL_INPUTS + DIR_INPUTS)
    + ONE_BLOB_SIZE * ROUGHNESS_INPUTS
    + METALLIC_INPUTS
    + ALBEDO_INPUTS;
const ALBEDO_OFFSET: usize =
    POSITION_INPUTS + NOMRAL_INPUTS + DIR_INPUTS + ROUGHNESS_INPUTS + METALLIC_INPUTS;
// type FeatureMat = na::SMatrix<f32, { FEATURE_SIZE }, 5>;
type InputVec = na::SVector<f32, { INPUT_SIZE }>;

fn one_blob_encoding(input: f32, i: usize) -> f32 {
    assert!(0.0 <= input && input <= 1.0);
    let sigma = 1.0 / ONE_BLOB_SIZE as f32;
    let x = (i as f32 / ONE_BLOB_SIZE as f32 - input) as f32;
    1.0 / ((2.0 * std::f32::consts::PI).sqrt() * sigma) * (-x * x / (2.0 * sigma * sigma)).exp()
    // input
}

fn nrc_encoding(v: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    let mut u: na::DMatrix<f32> = na::DMatrix::zeros(MAPPED_SIZE, v.ncols());
    assert!(v.nrows() == INPUT_SIZE);
    for c in 0..v.ncols() {
        for i in 0..POSITION_INPUTS {
            for f in 0..POSITION_FREQ {
                if f > 0 {
                    u[(i * POSITION_FREQ + f, c)] =
                        (v[(i, c)] * 2.0f32.powi((f - 1) as i32) * PI).sin();
                } else {
                    u[(i * POSITION_FREQ + f, c)] = v[(i, c)];
                }
            }
        }
        let offset_u = POSITION_INPUTS * POSITION_FREQ;
        let offset_v = POSITION_INPUTS;
        for i in 0..NOMRAL_INPUTS {
            for k in 0..ONE_BLOB_SIZE {
                u[(offset_u + i * ONE_BLOB_SIZE + k, c)] =
                    one_blob_encoding(v[(offset_v + i, c)], k);
            }
        }
        let offset_u = offset_u + NOMRAL_INPUTS * ONE_BLOB_SIZE;
        let offset_v = offset_v + NOMRAL_INPUTS;
        for i in 0..DIR_INPUTS {
            for k in 0..ONE_BLOB_SIZE {
                u[(offset_u + i * ONE_BLOB_SIZE + k, c)] =
                    one_blob_encoding(v[(offset_v + i, c)], k);
            }
        }
        let offset_u = offset_u + DIR_INPUTS * ONE_BLOB_SIZE;
        let offset_v = offset_v + DIR_INPUTS;
        for k in 0..ONE_BLOB_SIZE {
            u[(offset_u + k, c)] = one_blob_encoding(1.0 - (-v[(offset_v, c)]).exp(), k);
        }
        let offset_u = offset_u + ROUGHNESS_INPUTS * ONE_BLOB_SIZE;
        let offset_v = offset_v + ROUGHNESS_INPUTS;
        for k in 0..(METALLIC_INPUTS + ALBEDO_INPUTS) {
            u[(offset_u + k, c)] = v[(offset_v + k, c)];
        }
        assert!(offset_u + METALLIC_INPUTS + ALBEDO_INPUTS == MAPPED_SIZE);
        assert!(offset_v + METALLIC_INPUTS + ALBEDO_INPUTS == INPUT_SIZE);
    }
    assert!(u.nrows() == MAPPED_SIZE);
    u
}
// position_encoding_func_v3!(nrc_encoding, INPUT_SIZE, FEATURE_SIZE, MAX_FREQ);

fn sph(v: Vec3) -> Vec2 {
    let v = dir_to_spherical(v);
    vec2(v.x / PI, v.y / (2.0 * PI))
}

fn tone_mapping(x: f32) -> f32 {
    x / (x + 1.0)
}
fn inv_tone_mapping(x: f32) -> f32 {
    x / (1.0 - x)
}
struct TrainRecord {
    queue: Vec<f32>,
    target: Vec<f32>,
}
struct RadianceCache {
    model: Arc<MLP>,
    bound: Bounds3f,
    query_queue: RwLock<Vec<f32>>,
    query_result: RwLock<na::DMatrix<f32>>,
    train: RwLock<TrainRecord>,
}
impl Clone for RadianceCache {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            bound: self.bound,
            query_result: RwLock::new(na::DMatrix::zeros(0, 0)),
            query_queue: RwLock::new(vec![]),
            train: RwLock::new({
                TrainRecord {
                    target: vec![],
                    queue: vec![],
                }
            }),
        }
    }
}
#[derive(Clone, Copy)]
struct QueryRecord {
    n: Vec2,
    info: BsdfInfo,
    x: Vec3,
    dir: Vec2,
}

#[derive(Clone, Copy)]
struct PathState {
    beta: SampledSpectrum,
    li: SampledSpectrum,
    query_index: Option<usize>,
    thread_index: usize,
}
impl RadianceCache {
    fn new(model: MLP) -> Self {
        Self {
            model: Arc::new(model),
            bound: Bounds3f {
                min: (vec3(1.0, 1.0, 1.0) * -500.0).into(),
                max: (vec3(1.0, 1.0, 1.0) * 500.0).into(),
            },
            query_queue: RwLock::new(vec![]),
            query_result: RwLock::new(na::DMatrix::zeros(0, 0)),
            train: RwLock::new(TrainRecord {
                queue: vec![],
                target: vec![],
            }),
        }
    }
    fn get_input_vec(r: &QueryRecord) -> InputVec {
        InputVec::from_iterator(
            [r.x.x, r.x.y, r.x.z]
                .iter()
                .map(|x| *x)
                .chain([r.n.x, r.n.y].iter().map(|x| *x))
                .chain([r.dir.x, r.dir.y].iter().map(|x| *x))
                .chain(once(r.info.roughness))
                .chain(once(r.info.metallic))
                .chain(
                    [
                        r.info.albedo.samples.x,
                        r.info.albedo.samples.y,
                        r.info.albedo.samples.z,
                    ]
                    .iter()
                    .map(|x| *x),
                ),
        )
    }
    // returns index to query_result
    fn record_infer(&self, r: &QueryRecord) -> Option<usize> {
        if !self.bound.contains(r.x) {
            return None;
        }
        let v = Self::get_input_vec(r);
        let mut queue = self.query_queue.write();
        let idx = queue.len() / INPUT_SIZE;
        queue.extend(v.iter());
        // let s: SampledSpectrum = self.model.infer(&nrc_encoding(&v)).into();
        Some(idx)
    }
    fn infer_single(&self, r: &QueryRecord) -> Option<SampledSpectrum> {
        if !self.bound.contains(r.x) {
            return None;
        }
        let v = Self::get_input_vec(r);
        let mut r = self
            .model
            .infer(nrc_encoding(&na::DMatrix::from_column_slice(
                v.nrows(),
                1,
                v.as_slice(),
            )));
        r.iter_mut().for_each(|x| *x = inv_tone_mapping(*x));
        assert!(r.nrows() == SampledSpectrum::N_SAMPLES && r.ncols() == 1);
        let s: na::SVector<f32, 3> = na::SVector::from_iterator(r.iter().map(|x| *x as f32));
        Some(SampledSpectrum {
            samples: Vec3A::from([s[0], s[1], s[2]]),
        })
    }
    fn record_train(&self, r: &QueryRecord, target: &SampledSpectrum) {
        if !self.bound.contains(r.x) {
            return;
        }
        let v = Self::get_input_vec(r);
        let mut train = self.train.write();
        assert_eq!(
            train.queue.len() / INPUT_SIZE,
            train.target.len() / SampledSpectrum::N_SAMPLES,
            "train record lens diff! {} and {}",
            train.queue.len() / INPUT_SIZE,
            train.target.len() / SampledSpectrum::N_SAMPLES
        );
        train.queue.extend(v.iter());
        let albedo = r.info.albedo.samples;
        let albedo = [albedo.x, albedo.y, albedo.z];
        let target = [target.samples.x, target.samples.y, target.samples.z];
        let target = target
            .iter()
            .zip(albedo.iter())
            .map(|(a, b)| if *b != 0.0 { *a / *b } else { 0.0 });
        train.target.extend(target.map(|x| tone_mapping(x as f32)));
    }
    fn infer(&self) {
        {
            let queue = self.query_queue.write();
            assert!(queue.len() % INPUT_SIZE == 0);
            let mut inputs = na::DMatrix::<f32>::from_column_slice(
                INPUT_SIZE,
                queue.len() / INPUT_SIZE,
                &queue[..],
            );
            inputs = nrc_encoding(&inputs);
            let mut result = self.query_result.write();
            *result = {
                let mut r = self.model.infer(inputs);
                r.iter_mut().for_each(|x| *x = inv_tone_mapping(*x));
                for i in 0..r.ncols() {
                    for j in 0..r.nrows() {
                        r[(j, i)] *= queue[i * INPUT_SIZE + ALBEDO_OFFSET + j];
                    }
                }
                r
            };
        }
        {
            let mut queue = self.query_queue.write();
            queue.clear();
        }
    }
    fn train(&mut self) -> f32 {
        let mut train = self.train.write();
        assert!(train.queue.len() % INPUT_SIZE == 0);
        assert!(train.target.len() % SampledSpectrum::N_SAMPLES == 0);
        let queue = &train.queue;
        let mut inputs =
            na::DMatrix::<f32>::from_column_slice(INPUT_SIZE, queue.len() / INPUT_SIZE, &queue[..]);
        inputs = nrc_encoding(&inputs);
        let targets = na::DMatrix::<f32>::from_column_slice(
            SampledSpectrum::N_SAMPLES,
            &train.target.len() / SampledSpectrum::N_SAMPLES,
            &train.target[..],
        );
        let loss = Arc::get_mut(&mut self.model)
            .unwrap()
            .train(inputs, &targets, Loss::RelativeL2);
        train.queue.clear();
        train.target.clear();
        loss
    }
}
pub struct CachedPathTracer {
    pub spp: u32,
    pub training_iters: u32,
    pub batch_size: u32,
    pub max_depth: u32,
    pub visualize_cache: bool,
    pub learning_rate: f32,
}
// impl Default for CachedPathTracer {
//     fn default() -> Self {
//         Self {
//             spp: 32,
//             training_iters: 1024,
//             batch_size: 1024,
//             max_depth: 5,
//             visualize_cache: false,
//         }
//     }
// }
#[derive(Copy, Clone)]
struct Vertex {
    x: Vec3,
    dir: Vec3,
    info: BsdfInfo,
    n: Vec3,
    radiance: SampledSpectrum,
}
#[derive(Copy, Clone)]
struct VertexTemp {
    beta: SampledSpectrum,
    li: SampledSpectrum,
}
fn mis_weight(mut pdf_a: f32, mut pdf_b: f32) -> f32 {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    pdf_a / (pdf_a + pdf_b)
}

impl CachedPathTracer {
    fn li<'a>(
        &self,
        scene: &'a Scene,
        mut ray: Ray,
        sampler: &mut dyn Sampler,
        max_depth: u32,
        path: &mut Vec<Vertex>,
        path_tmp: &mut Vec<VertexTemp>,
        training: bool,
        enable_cache: bool,
        use_cache_after: u32,
        cache: &RadianceCache,
        arena: &Bump,
    ) -> PathState {
        let mut li = SampledSpectrum::zero();
        let mut beta = SampledSpectrum::one();
        let mut prev_n: Option<Vec3> = None;
        let mut prev_bsdf_pdf: Option<f32> = None;
        let terminatin_coeff = 0.01;
        let mut depth = 0;
        path_tmp.clear();
        path.clear();
        // if training {
        //     path_tmp.push(VertexTemp {
        //         beta: SampledSpectrum::one(),
        //         li: SampledSpectrum::zero(),
        //     });
        //     path.push(Vertex {
        //         x: ray.o,
        //         dir: dir_to_spherical(ray.d),
        //         radiance: SampledSpectrum::zero(),
        //     });
        // }

        macro_rules! accumulate_radiance {
            ($rad:expr) => {{
                li += beta * $rad;
                if training {
                    for i in 0..path_tmp.len() {
                        let beta = path_tmp[i].beta;
                        path_tmp[i].li += beta * $rad;
                    }
                }
            }};
        }
        macro_rules! accumulate_beta {
            ($b:expr) => {{
                beta *= $b;
                if training {
                    for i in 0..path_tmp.len() {
                        path_tmp[i].beta *= $b;
                    }
                }
            }};
        }
        // let accumulate_beta = {
        //     // let p_li = &mut li as *mut SampledSpectrum;
        //     let p_beta = &mut beta as *mut SampledSpectrum;
        //     let p_path_tmp = path_tmp as *mut Vec<VertexTemp>;
        //     move |b: SampledSpectrum| unsafe {
        //         // *p_li += *p_beta * l;
        //         *p_beta *= b;
        //         let path_tmp = &mut *p_path_tmp;
        //         if training {
        //             for i in 0..path_tmp.len() {
        //                 path_tmp[i].beta *= b;
        //             }
        //         }
        //     }
        // };
        let mut termination_a0: f32 = 0.0;
        let mut termination_a: f32 = 0.0;
        let mut prev_x: Vec3 = ray.o;
        let mut prev_pdf: f32 = 0.0;
        loop {
            if let Some(si) = scene.intersect(&ray) {
                let ng = si.ng;
                let shape = si.shape;
                let opt_bsdf = si.evaluate_bsdf(arena);
                if opt_bsdf.is_none() {
                    break;
                }
                let p = ray.at(si.t);
                if depth == 0 {
                    let w = (ray.o - p).normalize();
                    termination_a0 = (ray.o - p).length_squared() / (4.0 * PI * w.dot(ng).abs());
                }
                if depth > 0 {
                    let w = (prev_x - p).normalize();
                    termination_a +=
                        ((prev_x - p).length_squared() / (prev_pdf * w.dot(ng).abs())).sqrt();
                }
                let bsdf = opt_bsdf.unwrap();
                if let Some(light) = scene.get_light_of_shape(shape) {
                    // li += beta * light.le(&ray);
                    if depth == 0 {
                        accumulate_radiance!(light.le(&ray));
                    } else {
                        let light_pdf = scene.light_distr.pdf(light)
                            * light
                                .pdf_li(
                                    ray.d,
                                    &ReferencePoint {
                                        p: ray.o,
                                        n: prev_n.unwrap(),
                                    },
                                )
                                .1;
                        let bsdf_pdf = prev_bsdf_pdf.unwrap();
                        assert!(light_pdf.is_finite());
                        assert!(light_pdf >= 0.0);
                        let weight = mis_weight(bsdf_pdf, light_pdf);
                        accumulate_radiance!(light.le(&ray) * weight);
                    }
                }
                if training && depth <= use_cache_after {
                    path_tmp.push(VertexTemp {
                        beta: SampledSpectrum::one(),
                        li: SampledSpectrum::zero(),
                    });
                    path.push(Vertex {
                        x: p,
                        n: ng,
                        info: bsdf.info(),
                        dir: ray.d,
                        radiance: SampledSpectrum::zero(),
                    });
                }
                {
                    let cache_enable_depth = if training {
                        use_cache_after + 2
                    } else {
                        use_cache_after
                    };

                    if enable_cache && depth >= cache_enable_depth {
                        let record = QueryRecord {
                            x: p,
                            dir: sph(ray.d),
                            info: bsdf.info(),
                            n: sph(ng),
                        };
                        if training {
                            if let Some(r) = cache.infer_single(&record) {
                                accumulate_radiance!(r);
                                while path_tmp.len() >= use_cache_after as usize {
                                    path_tmp.pop();
                                    path.pop();
                                }
                                break;
                            }
                        } else if termination_a * termination_a > terminatin_coeff * termination_a0
                        {
                            if let Some(idx) = cache.record_infer(&record) {
                                // accumulate_radiance(radiance);
                                return PathState {
                                    beta,
                                    li,
                                    query_index: Some(idx),
                                    thread_index: rayon::current_thread_index().unwrap_or(0),
                                };
                            }
                        }
                    }
                }

                let wo = -ray.d;
                if depth >= max_depth {
                    break;
                }
                depth += 1;
                {
                    let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());

                    let sample_self = if let Some(light2) = scene.get_light_of_shape(shape) {
                        if light as *const dyn Light == light2 as *const dyn Light {
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if !sample_self {
                        let p_ref = ReferencePoint { p, n: ng };
                        let light_sample = light.sample_li(sampler.next3d(), &p_ref);
                        let light_pdf = light_sample.pdf * light_pdf;
                        if !light_sample.li.is_black() && !scene.occlude(&light_sample.shadow_ray) {
                            let bsdf_pdf = bsdf.evaluate_pdf(wo, light_sample.wi);
                            let weight = mis_weight(light_pdf, bsdf_pdf);
                            accumulate_radiance!(
                                bsdf.evaluate(wo, light_sample.wi)
                                    * ng.dot(light_sample.wi).abs()
                                    * light_sample.li
                                    / light_pdf
                                    * weight
                            );
                        }
                    }
                }

                if let Some(bsdf_sample) = bsdf.sample(sampler.next2d(), wo) {
                    let wi = bsdf_sample.wi;
                    ray = Ray::spawn(p, wi).offset_along_normal(ng);
                    accumulate_beta!(bsdf_sample.f * wi.dot(ng).abs() / bsdf_sample.pdf);
                    prev_x = p;
                    prev_pdf = bsdf_sample.pdf;
                    prev_bsdf_pdf = Some(bsdf_sample.pdf);
                    prev_n = Some(si.ng);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if training {
            for i in 0..path.len() {
                path[i].radiance = path_tmp[i].li;
            }
        }
        PathState {
            beta,
            li,
            query_index: None,
            thread_index: rayon::current_thread_index().unwrap_or(0),
        }
    }
}
#[derive(Clone)]
struct PerThreadData {
    path: Vec<Vertex>,
    path_tmp: Vec<VertexTemp>,
}
impl Integrator for CachedPathTracer {
    fn render(&self, scene: &Scene) -> Film {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        let opt_params = AdamParams {
            learning_rate: self.learning_rate,
            ..Default::default()
        };
        let opt = Adam::new(opt_params);
        let model = {
            let layers = vec![
                Layer::new(Some(Activation::Relu), MAPPED_SIZE, 64, true),
                Layer::new(Some(Activation::Relu), 64, 64, false),
                Layer::new(Some(Activation::Relu), 64, 64, false),
                Layer::new(Some(Activation::Relu), 64, 64, false),
                Layer::new(Some(Activation::Relu), 64, 64, false),
                Layer::new(Some(Activation::Relu), 64, 64, false),
                Layer::new(Some(Activation::Relu), 64, 64, false),
                Layer::new(Some(Activation::Sigmoid), 64, SampledSpectrum::N_SAMPLES, false),
            ];
            MLP::new(layers, opt)
        };
        let mut cache = RadianceCache::new(model);
        let mut samplers: Vec<Box<dyn Sampler>> = vec![];
        for i in 0..npixels {
            samplers.push(Box::new(SobolSampler::new(i as u64)));
        }
        let mut states = vec![
            PathState {
                beta: SampledSpectrum::one(),
                li: SampledSpectrum::zero(),
                query_index: None,
                thread_index: 0,
            };
            npixels
        ];
        let mut per_thread_data: Vec<PerThreadData> = vec![
            PerThreadData {
                path: vec![],
                path_tmp: vec![]
            };
            rayon::current_num_threads()
        ];
        let arenas = PerThread::new(|| Bump::new());
        let training_freq = npixels as f64 / self.batch_size as f64;
        for iter in 0..self.training_iters {
            let now = std::time::Instant::now();
            let p_samplers = &UnsafePointer::new(&mut samplers as *mut Vec<Box<dyn Sampler>>);
            let p_per_thread_data =
                &UnsafePointer::new(&mut per_thread_data as *mut Vec<PerThreadData>);
            let mut rng = rand::thread_rng();
            let training_pixels = {
                let mut set = HashSet::new();
                let mut training_pixels = vec![];
                for _ in 0..self.batch_size {
                    let i = loop {
                        let i: usize = rng.gen::<usize>() % npixels;
                        if !set.contains(&i) {
                            break i;
                        }
                    };
                    set.insert(i);
                    training_pixels.push(i);
                }
                training_pixels
            };
            {
                parallel_for(self.batch_size as usize, 64, |id| {
                    let id = training_pixels[id];
                    let samplers = unsafe { p_samplers.as_mut().unwrap() };
                    let thread_data = unsafe {
                        &mut (p_per_thread_data.as_mut().unwrap())
                            [rayon::current_thread_index().unwrap()]
                    };
                    let sampler = &mut samplers[id];
                    sampler.start_next_sample();
                    // let mut sampler = PCGSampler { rng: Pcg::new(id) };
                    let x = (id as u32) % scene.camera.resolution().x;
                    let y = (id as u32) / scene.camera.resolution().x;
                    let pixel = uvec2(x, y);
                    let (ray, _ray_weight) = scene.camera.generate_ray(pixel, sampler.as_mut());
                    let path = &mut thread_data.path;
                    let path_tmp = &mut thread_data.path_tmp;
                    // let mut rng = rand::thread_rng();
                    // let training = ((x + iter) % training_freq == 0)
                    // && ((y + iter / training_freq) % training_freq == 0);
                    let training = true; //rng.gen_bool(1.0 / training_freq);
                    if !training {
                        return;
                    }
                    let arena = arenas.get_mut();
                    let _li = self.li(
                        scene,
                        ray,
                        sampler.as_mut(),
                        self.max_depth,
                        path,
                        path_tmp,
                        training,
                        false,
                        1,
                        &cache,
                        arena,
                    );
                    if training {
                        for vertex in path.iter() {
                            // vertex.dir.into_iter().for_each(|x| assert!(!x.is_nan()));
                            let record = QueryRecord {
                                x: vertex.x,
                                dir: sph(vertex.dir),
                                info: vertex.info,
                                n: sph(vertex.n),
                            };
                            cache.record_train(&record, &vertex.radiance);
                        }
                    }
                    arena.reset();
                });
            }
            {
                // println!(
                //     "batch size:{} {}",
                //     cache.train.read().queue.len() / INPUT_SIZE,
                //     npixels as f64 * 1.0 / training_freq
                // );
                let loss = cache.train();
                if iter % 20 == 0 {
                    log::info!(
                        "training pass {} finished in {}s, loss = {}",
                        iter,
                        now.elapsed().as_secs_f32(),
                        loss
                    );
                }
            }
        }

        // println!("visiualizing");
        for iter in 0..self.spp {
            let trained_caches = vec![cache.clone(); rayon::current_num_threads()];
            let mut rt_time = 0.0;
            let mut infer_time = 0.0;
            let p_samplers = &UnsafePointer::new(&mut samplers as *mut Vec<Box<dyn Sampler>>);
            let p_per_thread_data =
                &UnsafePointer::new(&mut per_thread_data as *mut Vec<PerThreadData>);
            let p_states = &UnsafePointer::new(&mut states as *mut Vec<PathState>);
            let now = std::time::Instant::now();
            let chunk_size = 512 * 512;
            let mut chunk_offset = 0;
            while chunk_offset < npixels {
                rt_time += profile_fn(|| {
                    parallel_for(chunk_size.min(npixels - chunk_offset), 256, |id| {
                        let id = id + chunk_offset;
                        let samplers = unsafe { p_samplers.as_mut().unwrap() };
                        let states = unsafe { p_states.as_mut().unwrap() };
                        let thread_data = unsafe {
                            &mut (p_per_thread_data.as_mut().unwrap())
                                [rayon::current_thread_index().unwrap()]
                        };
                        let trained_cache = &trained_caches[rayon::current_thread_index().unwrap()];
                        let state = &mut states[id];
                        let sampler = &mut samplers[id];
                        sampler.start_next_sample();
                        // let mut sampler = PCGSampler { rng: Pcg::new(id) };
                        let x = (id as u32) % scene.camera.resolution().x;
                        let y = (id as u32) / scene.camera.resolution().x;
                        let pixel = uvec2(x, y);
                        let (ray, _ray_weight) = scene.camera.generate_ray(pixel, sampler.as_mut());
                        let path = &mut thread_data.path;
                        let path_tmp = &mut thread_data.path_tmp;
                        let mut rng = rand::thread_rng();
                        let training = rng.gen_bool(1.0 / training_freq);
                        let arena = arenas.get_mut();
                        *state = self.li(
                            scene,
                            ray,
                            sampler.as_mut(),
                            self.max_depth,
                            path,
                            path_tmp,
                            training,
                            !training,
                            if self.visualize_cache { 0 } else { 1 },
                            &trained_cache,
                            arena,
                        );
                        arena.reset();
                        if training {
                            for vertex in path.iter() {
                                // vertex.dir.into_iter().for_each(|x| assert!(!x.is_nan()));
                                let record = QueryRecord {
                                    x: vertex.x,
                                    dir: sph(vertex.dir),
                                    info: vertex.info,
                                    n: sph(vertex.n),
                                };
                                cache.record_train(&record, &vertex.radiance);
                            }
                        }
                    });
                })
                .1;
                infer_time += profile_fn(|| {
                    parallel_for(trained_caches.len(), 1, |i| {
                        trained_caches[i].infer();
                    });
                })
                .1;
                parallel_for(chunk_size.min(npixels - chunk_offset), 256, |id| {
                    let id = id + chunk_offset;
                    let x = (id as u32) % scene.camera.resolution().x;
                    let y = (id as u32) / scene.camera.resolution().x;
                    let pixel = uvec2(x, y);
                    let state = &states[id];
                    let cache = &trained_caches[state.thread_index];
                    let li = {
                        if let Some(idx) = state.query_index {
                            let results = cache.query_result.read();
                            let s: na::SVector<f32, 3> = na::SVector::from_iterator(
                                results.column(idx).iter().map(|x| *x as f32),
                            );
                            let result = SampledSpectrum {
                                samples: Vec3A::from([s[0], s[1], s[2]]),
                            };
                            state.li + state.beta * result
                        } else {
                            state.li
                        }
                    };
                    film.add_sample(pixel, li, 1.0);
                });

                chunk_offset += chunk_size;
            }
            log::info!(
                "rendering pass {} finished in {}s: ray trace {}s, infer {}s",
                iter,
                now.elapsed().as_secs_f64(),
                rt_time,
                infer_time,
            );
            let mut trained_caches = trained_caches;
            trained_caches.clear();
            cache.train();
        }
        // println!("visiualizing");
        // parallel_for(npixels, 256, |id| {
        //     let mut sampler = PCGSampler { rng: Pcg::new(id) };
        //     let x = (id as u32) % scene.camera.resolution().x;
        //     let y = (id as u32) / scene.camera.resolution().x;
        //     let pixel = uvec2(x, y);
        //     let (ray, _ray_weight) = scene.camera.generate_ray(pixel, &mut sampler);
        //     let lk = cache.read();
        //     let cache = &*lk;
        //     // let li = cache.infer(&ray.o, &dir_to_spherical(ray.d));
        //     let mut li = SampledSpectrum::zero();
        //     if let Some(si) = scene.intersect(&ray) {
        //         let p = ray.o; //ray.at(si.t * 0.9);
        //         li = cache.infer(&p, &dir_to_spherical(ray.d));
        //         // let n = si.ng;
        //         // let frame = Frame::from_normal(n);
        //         // f
        //         // println!("{}", li.samples);
        //     }
        //     film.add_sample(uvec2(x, y), &li, 1.0);
        // });
        film
    }
}
