use crate::{serde_json, FilmTile};
use crate::{Bounds2u, Pixel};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
pub type Id = usize;
pub type JobId = usize;
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct FileMetadata {
    pub path: String,
    pub size: u64,
    pub last_modified: Duration,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ReplicaInit {
    pub dependencies: Vec<FileMetadata>,
    pub integrator_settings: Value,
    pub work_settings: Value,
    pub id: Id,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum WorkItem {
    Block(Bounds2u),
    Chains(u64),
    Spp(u64),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ToReplica {
    ReplicaInit(ReplicaInit),
    GetFileResult { len: u64, err: Option<String> },
    GetWorkItemResult(Option<WorkItem>),
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ToPrimary {
    Register,
    GetFile { path: String },
    Heartbeat { seconds: u64 },
    GetWorkItem,
    SyncFilm(FilmTile),
}
