use crate::{*, camera::*, film::*, scene::*};

pub trait Integrator {
    fn render(&self, scene: &Scene, film: &mut Film);
}

pub mod pt;
pub mod normal;
pub mod mcmc;