use std::{sync::Arc, time::Instant};

use crate::{
    color::{Color, ColorRepr, FlatColor},
    film::*,
    integrator::pt::{self, PathTracer},
    sampler::*,
    scene::*,
    *,
};
use serde::{Deserialize, Serialize};

use super::{Integrator, RenderOptions};
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub seed: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            spp: 256,
            max_depth: 7,
            rr_depth: 5,
            spp_per_pass: 64,
            use_nee: true,
            indirect_only: false,
            seed: 0,
        }
    }
}
#[derive(Clone)]
pub struct GradientPathTracer {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub seed: u64,
    config: Config,
}

impl GradientPathTracer {
    pub fn new(device: Device, config: Config) -> Self {
        Self {
            device,
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.indirect_only,
            seed: config.seed,
            config,
        }
    }
}
impl Integrator for GradientPathTracer {
    #[allow(non_snake_case)]
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_repr: ColorRepr,
        film: &mut Film,
        _options: &RenderOptions,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig:{:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        let npixels = resolution.x as usize * resolution.y as usize;
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let evaluators = scene.evaluators(color_repr);
        let pt = PathTracer::new(
            self.device.clone(),
            pt::Config {
                max_depth: self.max_depth,
                spp_per_pass: self.spp_per_pass,
                use_nee: self.use_nee,
                rr_depth: self.rr_depth,
                indirect_only: self.indirect_only,
                pixel_offset: [0, 0],
                spp: self.spp,
            },
        );
        let filter = film.filter();
        let primal = film;
        let Gx = primal.clone();
        let Gy = primal.clone();
        let sampler_creator = sampler_config.creator(self.device.clone(), &scene, self.spp);
        let kernel = self
            .device
            .create_kernel::<(u32,)>(&|spp_per_pass: Expr<u32>| {
                if !is_cpu_backend() {
                    set_block_size([8, 8, 1]);
                }

                let p = dispatch_id().xy();
                for_range(const_(0)..spp_per_pass.int(), |_| {

                    let ip = p.int();
                    let colors = var!([FlatColor; 5]);
                    let weights = var!([f32; 5]);
                    let offsets = [
                        Int2::new(0, 0),
                        Int2::new(1, 0),
                        Int2::new(0, 1),
                        Int2::new(-1, 0),
                        Int2::new(0, -1),
                    ];
                    let offsets = const_(offsets);

                    for_range(0..5u32, |i| {
                        let sampler = sampler_creator.create(p);
                        sampler.start();
                        let offset = offsets.read(i);
                        let shifted = ip + offset;
                        if_!(
                            !(shifted.cmplt(0).any()
                                | shifted.cmpge(const_(resolution).int()).any()),
                            {
                                let shifted = shifted.uint();
                                let (ray, ray_color, ray_w) = scene.camera.generate_ray(
                                    filter,
                                    shifted,
                                    sampler.as_ref(),
                                    color_repr,
                                );
                                let l = pt.radiance(&scene, ray, sampler.as_ref(), &evaluators)
                                    * ray_color;
                                colors.write(i, l.flatten());
                                weights.write(i, ray_w);
                            }
                        );
                        if_!(i.cmpne(4), {
                            sampler.forget();
                        });
                    });
                    let base = Color::from_flat(color_repr, colors.read(0));
                    let base_w = weights.read(0);

                    let x_p1_y = Color::from_flat(color_repr, colors.read(1));
                    // let x_p1_y_w = weights.read(1);
                    let x_y_p1 = Color::from_flat(color_repr, colors.read(2));
                    // let x_y_p1_w = weights.read(2);
                    let x_m1_y = Color::from_flat(color_repr, colors.read(3));
                    // let x_m1_y_w = weights.read(3);
                    let x_y_m1 = Color::from_flat(color_repr, colors.read(4));
                    // let x_y_m1_w = weights.read(4);

                    if_!(ip.x().cmpgt(0), {
                        Gx.add_sample(p.float() - make_float2(1.0, 0.0), &(base - x_m1_y), base_w);
                    });
                    if_!(ip.y().cmpgt(0), {
                        Gy.add_sample(p.float() - make_float2(0.0, 1.0), &(base - x_y_m1), base_w);
                    });
                    Gx.add_sample(p.float(), &(x_p1_y - base), base_w);
                    Gy.add_sample(p.float(), &(x_y_p1 - base), base_w);
                    for_range(0..5u32, |i| {
                        let offset = offsets.read(i);
                        let shifted = ip + offset;
                        if_!(
                            !(shifted.cmplt(0).any()
                                | shifted.cmpge(const_(resolution).int()).any()),
                            {
                                // let num_samples = var!(u32, 5);
                                // if_!(shifted.x().cmpeq(0), {
                                //     *num_samples.get_mut() -= 1;
                                // });
                                // if_!(shifted.y().cmpeq(0), {
                                //     *num_samples.get_mut() -= 1;
                                // });
                                // if_!(shifted.x().cmpeq(const_(resolution.x).int() - 1), {
                                //     *num_samples.get_mut() -= 1;
                                // });
                                // if_!(shifted.y().cmpeq(const_(resolution.y).int() - 1), {
                                //     *num_samples.get_mut() -= 1;
                                // });
                                primal.add_sample(
                                    shifted.float(),
                                    &Color::from_flat(color_repr, colors.read(i)),
                                    weights.read(i),
                                );
                            }
                        );
                    });
                });
            });
        let stream = self.device.default_stream();
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;
        stream.with_scope(|s| {
            while cnt < self.spp {
                let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
                let mut cmds = vec![];
                let tic = Instant::now();
                cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1], &cur_pass));
                s.submit(cmds);
                s.synchronize();
                let toc = Instant::now();
                acc_time += toc.duration_since(tic).as_secs_f64();
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
            }
        });
        progress.finish();
        log::info!("Rendering finished in {:.2}s", acc_time);
    }
}
pub fn render(
    device: Device,
    scene: Arc<Scene>,sampler: SamplerConfig,
    color_repr: ColorRepr,
    film: &mut Film,
    config: &Config,
    options: &RenderOptions,
) {
    let gpt = GradientPathTracer::new(device.clone(), config.clone());
    gpt.render(scene, sampler, color_repr, film, options);
}
