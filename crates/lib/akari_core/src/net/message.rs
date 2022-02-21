use crate::serde_json;
use crate::{Bounds2u, Pixel};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
pub type Id = usize;
pub type JobId = usize;
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct FileRecord {
    pub path: String,
    pub size: u64,
    pub last_modified: Duration,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RemoteInit {
    pub dependencies: Vec<FileRecord>,
    pub integrator_settings: Value,
    pub work_settings: Value,
    pub id: Id,
}

// Reserved.
// #[derive(Clone, Serialize, Deserialize, Debug)]
// pub struct PeerMessage {
//     from: Id,
//     to: Id,
// }

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum LocalToRemote {
    Init(RemoteInit),
    FileTransferComplete {
        from: String,
        to: String,
    },
    /**
     * Upon receiving this message on remote,
     * a spmd::Scheduler instance should be created
     */
    ConfirmStart {
        id: Id,
        node_id: usize,
    },
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum RemoteToLocal {
    RequestWork { id: Id },
    RequestFileTransfer { from: String, to: String },
    RequestStart { id: Id },
    RequestShutdown { id: Id },
}
