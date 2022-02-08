pub mod api;
pub use akari_common::*;
pub use akari_core::*;
pub use akari_integrators as integrator;
pub use akari_utils::*;

pub struct Config {
    pub num_threads: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
        }
    }
}

pub fn init(config: Config) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads)
        .build_global()
        .unwrap();
}
