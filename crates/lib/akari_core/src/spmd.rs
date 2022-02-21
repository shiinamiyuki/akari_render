use bson::Bson;
use serde::{Deserialize, Serialize};

pub const PRIMARY_NODE_ID: usize = 0;
/** An SPMD scheduler
 *  Intended to schedule work across different nodes
 *  
 *  The actual work item is not stored in the scheduler.
 *  Rather only an id to the work item is stored, returned upon requests.
 *  Workers should reach an agreement as to what 1 unit of work item is.
 */
pub trait Scheduler {
    // should call this before requesting any work
    // all nodes need to call this at the same time
    // disagreement in count would cause panic
    fn set_total_work(&self, count: usize);
    fn num_nodes(&self) -> usize;
    fn total_work(&self) -> usize;

    // id of this node
    // 0 is the primary node
    fn node_id(&self) -> usize;
    /**
     * Returns Vec::new() when no further work is available
     */
    fn request_work(&self, count: usize) -> Vec<usize>;

    // // blocking send
    // fn send(&self, to: usize, msg: Message);

    // // blocking recv
    // fn recv(&self, from: usize) -> Message;

    fn broadcast(&self, message: Option<Message>) -> Option<Message>;

    fn gather(&self, message: Message) -> Vec<Message>;

    fn gather_all(&self, message: Message) -> Vec<Message>;

    fn barrier(&self);
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Message {
    pub tag: String,
    pub content: Bson,
}

impl dyn Scheduler {
    pub fn is_primary(&self) -> bool {
        self.node_id() == PRIMARY_NODE_ID
    }
    pub fn is_local_only(&self) -> bool {
        self.num_nodes() == 1
    }
}
