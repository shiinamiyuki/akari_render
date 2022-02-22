pub mod api;
use std::cell::RefCell;
use std::fs::File;
use std::io::{stdout, Stdout};

use akari_common::parking_lot::Mutex;
pub use akari_common::*;
pub use akari_core::*;
pub use akari_integrators as integrator;
pub use akari_utils::*;
use lazy_static::lazy_static;
use log::{Level, Metadata, Record};
use log::{LevelFilter, SetLoggerError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub num_threads: usize,
    pub log_output: String,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            log_output: "stdout".into(),
        }
    }
}

enum LogOutput {
    Stdout(Stdout),
    File(File),
}
struct SimpleLogger {
    output: Option<LogOutput>,
}
impl SimpleLogger {
    fn new() -> Self {
        Self { output: None }
    }
}

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

// lazy_static! {
//     static ref LOGGER: Mutex<SimpleLogger> = Mutex::new(SimpleLogger::new());
// }

// fn init_logger() -> Result<(), SetLoggerError> {
//     log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Info))
// }

pub fn init(config: Config) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_threads)
        .build_global()
        .unwrap();
    {
        // let mut logger = LOGGER.lock();
        let mut logger = SimpleLogger::new();
        match config.log_output {
            _x if _x == "stdout" => {
                logger.output = Some(LogOutput::Stdout(stdout()));
            }
            file @ _ => {
                logger.output = Some(LogOutput::File(
                    File::create(file).expect("failed to initialize logger"),
                ));
            }
        }
        // log::set_logger(logger)
        log::set_boxed_logger(Box::new(logger)).map(|()| log::set_max_level(LevelFilter::Info)).unwrap();
    }
}
