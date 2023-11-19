use std::cell::RefCell;

use crate::*;
use scene_graph::blender_util::{self, ImportMeshArgs};
use serde::*;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum SceneImportApi {
    Init,
    Finalize,

    ImportMesh(ImportMeshArgs),
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum SceneImportApiResult {
    None,
    Bool { value: bool },
}

pub struct SceneImportContext {
    scene: scene_graph::Scene,
}

thread_local! {
    static SCENE_IMPORT_CONTEXT: RefCell<Option<SceneImportContext>> = RefCell::new(None);
}
fn with_scene_import_context<T>(f: impl FnOnce(&mut SceneImportContext) -> T) -> T {
    SCENE_IMPORT_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        let ctx = ctx.as_mut().unwrap();
        f(ctx)
    })
}
pub fn handle_import_api(api: SceneImportApi) -> SceneImportApiResult {
    match api {
        SceneImportApi::Init => {
            SCENE_IMPORT_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(SceneImportContext {
                    scene: scene_graph::Scene::new(),
                });
            });
        }
        SceneImportApi::Finalize => {
            SCENE_IMPORT_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = None;
            });
        }
        SceneImportApi::ImportMesh(args) => {
            with_scene_import_context(|ctx| {
                blender_util::import_blender_mesh(&mut ctx.scene, args);
            });
        }
    }
    SceneImportApiResult::None
}
