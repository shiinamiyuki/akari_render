pub use crate::*;

#[pyclass]
pub struct SceneGraph {
    // scene: Scene,
}

#[pyclass]
pub struct Buffer {

}

#[pymethods]
impl SceneGraph {
    #[new]
    fn new() -> Self {
        Self {
            // scene: Scene::new(),
        }
    }

}

pub fn register_scenegraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SceneGraph>()?;
    Ok(())
}
