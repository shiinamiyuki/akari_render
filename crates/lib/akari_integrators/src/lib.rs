pub mod ao;
pub mod bdpt;
pub mod bidir;
pub mod erpt;
pub mod mmlt;
// pub mod nrc;
pub mod path;
// pub mod ppg;
pub mod pssmlt;
// pub mod sppm;
// pub mod spath;
// pub mod normalvis;
use crate::{film::Film, scene::Scene};
use akari_common::*;
use akari_core::*;
use akari_utils as util;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use util::{parallel_for, profile_fn, UnsafePointer};
use util::{AtomicFloat, RobustSum};

bitflags! {
    pub struct PartitionFlags: u8{
        const NONE = 0;
        const SPP = 1;
        const BLOCK = 2;
    }
}
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Partition {
    Spp(usize),
    Block(Bounds2u),
}
pub trait Integrator {
    fn render(&self, scene: &Scene) -> Film;
    fn support_block_rendering(&self) -> bool {
        false
    }
    // fn render_block(&mut self, scene: &Scene, bound: Bound2<u32>, film: &Film) {
    //     unimplemented!()
    // }
}
