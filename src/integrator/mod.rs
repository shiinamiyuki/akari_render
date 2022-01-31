pub mod ao;
pub mod bdpt;
pub mod pssmlt;
pub mod erpt;
pub mod mmlt;
pub mod nrc;
pub mod path;
pub mod ppg;
pub mod sppm;
pub mod spath;
pub mod normalvis;
use crate::{film::Film, scene::Scene};

pub trait Integrator {
    fn render(&mut self, scene: &Scene) -> Film;
    fn support_block_rendering(&self) -> bool {
        false
    }
    // fn render_block(&mut self, scene: &Scene, bound: Bound2<u32>, film: &Film) {
    //     unimplemented!()
    // }
}
