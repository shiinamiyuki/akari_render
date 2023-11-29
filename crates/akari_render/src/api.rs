use std::cell::RefCell;

use crate::*;
use scene_graph::{
    blender::{self, ImportMeshArgs},
    scene::Geometry,
    Buffer, Camera, Image, Instance, Material, NodeRef,
};
use serde::*;

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
pub enum SceneImportApi {
    Init { name: String },
    Finalize,
    ImportMesh { args: ImportMeshArgs },
    ImportMaterial { name: String, mat: Material },
    ImportInstance { name: String, instance: Instance },
    ImportCamera { camera: Camera },
    ImportBuffer { name: String, buffer: Buffer },
    WriteScene { path: String, compact: bool },
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
pub enum SceneImportApiResult {
    None,
    Bool { value: bool },
    Buffer { value: NodeRef<Buffer> },
    Geometry { value: NodeRef<Geometry> },
    Material { value: NodeRef<Material> },
    Instance { value: NodeRef<Instance> },
}

pub struct SceneImportContext {
    scene: scene_graph::Scene,
    name: String,
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

pub fn import(api: SceneImportApi) -> SceneImportApiResult {
    match api {
        SceneImportApi::Init { name } => {
            SCENE_IMPORT_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = Some(SceneImportContext {
                    scene: scene_graph::Scene::new(),
                    name,
                });
            });
        }
        SceneImportApi::Finalize => {
            SCENE_IMPORT_CONTEXT.with(|ctx| {
                *ctx.borrow_mut() = None;
            });
        }
        SceneImportApi::ImportBuffer { name, buffer } => {
            return with_scene_import_context(|ctx| SceneImportApiResult::Buffer {
                value: ctx.scene.add_buffer(Some(name), buffer),
            });
        }
        SceneImportApi::ImportMesh { args } => {
            return with_scene_import_context(|ctx| SceneImportApiResult::Geometry {
                value: blender::import_blender_mesh(&mut ctx.scene, args),
            });
        }
        SceneImportApi::ImportMaterial { name, mat } => {
            return with_scene_import_context(|ctx| SceneImportApiResult::Material {
                value: ctx.scene.add_material(Some(name), mat),
            });
        }
        SceneImportApi::ImportInstance { name, instance } => {
            return with_scene_import_context(|ctx| SceneImportApiResult::Instance {
                value: ctx.scene.add_instance(Some(name), instance),
            });
        }
        SceneImportApi::ImportCamera { camera } => {
            return with_scene_import_context(|ctx| {
                ctx.scene.camera = Some(camera);
                SceneImportApiResult::None
            });
        }
        SceneImportApi::WriteScene { path, compact } => {
            return with_scene_import_context(|ctx| {
                if compact {
                    unsafe { ctx.scene.compact(ctx.name.clone()).unwrap() };
                }
                ctx.scene.write_to_file(&path, !compact).unwrap();
                SceneImportApiResult::None
            });
        }
    }
    SceneImportApiResult::None
}
