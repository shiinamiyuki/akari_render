use crate::{*, camera::*, film::*, scene::*};

pub trait Integrator {
    fn render(&self, scene: &Scene, film: &mut Film)->luisa::Result<()>;
}

pub mod pt;
pub mod normal;