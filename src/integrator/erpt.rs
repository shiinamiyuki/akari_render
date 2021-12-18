use crate::distribution::Distribution1D;
use crate::integrator::path::PathTracer;
use crate::sampler::{PCGSampler, ReplaySampler, Sampler, PCG};
use crate::scene::Scene;
use crate::*;
use bidir::*;
use integrator::mmlt::*;
use integrator::*;

pub struct Erpt {
    pub spp: u32,
    pub direct_spp: u32,
    pub max_depth: u32,
    pub n_bootstrap: usize,
    pub mutations_per_chain: usize,
    pub n_chains: usize,
}
// impl Integrator for Erpt {
//     fn render(&mut self, scene: &Scene) -> Film {
//         log::info!("rendering direct lighting...");
//         let mut depth0_pt = PathTracer {
//             spp: self.direct_spp,
//             max_depth: 1,
//         };
//         let film_direct = depth0_pt.render(scene);
//         let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
//         log::info!("bootstrapping...");
//         let per_depth_e_avg: Vec<_> = (2..=self.max_depth)
//             .into_par_iter()
//             .map(|depth| MMLT::init_chain(self.n_bootstrap, 0, depth, scene).1)
//             .collect();
//         log::info!("average energy: {:?}", per_depth_e_avg);

//         let film = Film::new(&scene.camera.resolution());
//         log::info!("rendering {} spp", self.spp);
//         let chunks = (npixels + 255) / 256;
//         let progress = crate::util::create_progess_bar(chunks, "chunks");
//         parallel_for(npixels, 256, |id| {
//             let x = (id as u32) % scene.camera.resolution().x;
//             let y = (id as u32) / scene.camera.resolution().x;
//             let pixel = uvec2(x, y);
//             for _ in 0..self.spp{

//             }
//         });
//         progress.finish();
//     }
// }
