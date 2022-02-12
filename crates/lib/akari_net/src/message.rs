use std::time::Duration;

use akari::serde_json::Value;
use akari::{Bounds2u, Pixel};
use serde::{Deserialize, Serialize};
pub type Id = uuid::Uuid;
pub type JobId = usize;
#[derive(Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: String,
    pub size: u64,
    pub last_modified: Duration,
}
#[derive(Clone, Serialize, Deserialize)]
pub enum WorkerRequest {
    Init {
        id: Id,
    },
    Shutdown {
        id: Id,
    },
    PullJob {
        id: Id,
    },
    RequestScene {
        id: Id,
        job: JobId,
    },
    UpdateFilm {
        id: Id,
        bounds: Bounds2u,
        pixels: Vec<Pixel>,
    },
    RequestItem {
        id: Id,
        job: JobId,
    },
    HeartBeat(Id),
}

#[derive(Clone, Serialize, Deserialize)]
pub enum WorkerResponse {
    Nothing,
    WorkItems {
        job: Id,
        start: u64,
        end: u64,
    },
    NewJob {
        worker: Id,
        config: Value,
        scene: FileRecord,
        items: u64,
    },
    SendScene {
        worker: Id,
        filename: String,
        data: Vec<u8>, // a zip file
    },
    Abort {
        id: Id,
    },
}
#[derive(Clone, Serialize, Deserialize)]
pub enum ClientResponse {
    Started { id: JobId },
    Failed { id: JobId },
}
#[derive(Clone, Serialize, Deserialize)]
pub enum ClientRequest {
    NewJob { config: Value, scene: FileRecord },
    Pull { id: JobId },
    Abort {},
}
