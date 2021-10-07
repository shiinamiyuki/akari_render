pub mod ao;
pub mod bdpt;
pub mod nrc;
pub mod path;
pub mod sppm;
use crate::{film::Film, scene::Scene};

pub trait Integrator {
    fn render(&mut self, scene: &Scene) -> Film;
}
