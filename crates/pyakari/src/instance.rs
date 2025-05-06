pub use crate::*;
#[pyclass]
pub struct RendererInstance {}

#[pymethods]
impl RendererInstance {
    #[new]
    fn new(backend: &str) -> Self {
        Self {}
    }
}
pub fn register_renderer_instance(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RendererInstance>()?;
    Ok(())
}
