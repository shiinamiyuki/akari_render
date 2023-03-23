use luisa_compute::Polymorphic;

use crate::{camera::*, film::*, scene::*};

pub trait Integrator {
    fn render(&self, scene: &Scene, camera: &Polymorphic<(), dyn Camera>, film: &mut Film);
}

pub mod pt;
