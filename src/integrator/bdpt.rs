use crate::*;
use crate::bidir::*;
use crate::film::*;
use crate::integrator::*;
use crate::sampler::PCG;
use crate::sampler::PCGSampler;
pub struct Bdpt {
    pub spp: u32,
    pub max_depth: usize,
    pub debug: bool,
}
#[derive(Clone)]
struct BdptPerThreadData<'a> {
    scratch: Scratch<'a>,
    camera_path: Vec<Vertex<'a>>,
    light_path: Vec<Vertex<'a>>,
}
impl<'a> BdptPerThreadData<'a> {
    fn reset(&mut self) {
        self.scratch.new_eye_path.clear();
        self.scratch.new_light_path.clear();
        self.camera_path.clear();
        self.light_path.clear();
    }
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
        let chunks = (npixels + 255) / 256;
        let progress = crate::util::create_progess_bar(chunks, "chunks");
        let per_thread_data = util::PerThread::<BdptPerThreadData>::new(BdptPerThreadData {
            scratch: Scratch::new(),
            camera_path: vec![],
            light_path: vec![],
        });
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            let data = per_thread_data.get_mut();
            data.reset();
            let camera_path = &mut data.camera_path;
            let light_path = &mut data.light_path;
            let scratch = &mut data.scratch;
            let mut debug_acc = vec![];
            if self.debug {
                for _t in 2..=self.max_depth + 2 {
                    for _s in 0..=self.max_depth + 2 {
                        debug_acc.push(Spectrum::zero());
                    }
                }
            }
            for _ in 0..self.spp {
                bdpt::generate_camera_path(
                    scene,
                    &pixel,
                    &mut sampler,
                    self.max_depth + 2,
                    camera_path,
                );
                bdpt::generate_light_path(scene, &mut sampler, self.max_depth + 1, light_path);
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
                            light_path,
                            camera_path,
                            &mut sampler,
                            scratch,
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
            if (id + 1) % 256 == 0 {
                progress.inc(1);
            }
        });
        progress.finish();
        if self.debug {
            for t in 2..=(self.max_depth + 2) as isize {
                for s in 0..=(self.max_depth + 2) as isize {
                    let depth = s + t - 2;
                    if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                        continue;
                    }
                    let idx = get_index(s, t);
                    let film = &pyramid[idx];
                    film.write_exr(&format!("bdpt-d{}-s{}-t{}.exr", depth, s, t));
                }
            }
        }
        film
    }
}
