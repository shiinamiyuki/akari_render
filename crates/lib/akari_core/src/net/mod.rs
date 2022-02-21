use crate::spmd::Scheduler;

pub mod local;
pub mod message;
pub mod remote;

pub trait Initializer {
    fn finish(self) -> Box<dyn Scheduler>;
}
