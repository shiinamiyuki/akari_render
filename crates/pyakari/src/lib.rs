pub use pyo3::prelude::*;
mod scenegraph;
mod instance;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn pyakari(m: &Bound<'_, PyModule>) -> PyResult<()> {
    scenegraph::register_scenegraph(m)?;
    instance::register_renderer_instance(m)?;
    Ok(())
}
