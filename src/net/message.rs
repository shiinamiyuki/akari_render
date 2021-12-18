use bson::{doc, from_bson, to_bson, Document};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
async fn send_message_raw(stream: &mut TcpStream, bson: &Document) -> std::io::Result<()> {
    let mut buf: Vec<u8> = vec![];
    bson.to_writer(&mut buf).unwrap();
    let len = (buf.len() as u64).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&buf).await?;
    Ok(())
}

async fn recv_message_raw(stream: &mut TcpStream) -> bson::ser::Result<Document> {
    let mut len = [0u8; std::mem::size_of::<u64>()];
    stream.read_exact(&mut len).await?;
    let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
    stream.read_exact(&mut buf).await?;
    Ok(Document::from_reader(&mut buf.as_slice()).unwrap())
}
pub(crate) async fn send_message(stream: &mut TcpStream, message: &Message) -> std::io::Result<()> {
    let bson = to_bson(message).unwrap();
    send_message_raw(stream, bson.as_document().unwrap()).await
}

pub(crate) async fn recv_message(stream: &mut TcpStream) -> bson::de::Result<Message> {
    let bson = recv_message_raw(stream).await.unwrap();
    from_bson(bson::Bson::Document(bson))
}
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Worker,
    Client,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Init {
    pub(crate) role: Role,
    pub(crate) os: String,
    pub(crate) num_threads: u64,
    pub(crate) id: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransferTarget {
    pub(crate) from: String,
    pub(crate) to: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BinaryFileTransfer {
    pub(crate) target: TransferTarget,
    pub(crate) filename: String,
    pub(crate) content: Vec<u8>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextFileTransfer {
    pub(crate) target: TransferTarget,
    pub(crate) filename: String,
    pub(crate) content: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PollWorkerStatus {}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkerStatus {}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FileTransferRequest {
    pub(crate) filename: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArrayTransfer {
    pub(crate) target: TransferTarget,
    pub(crate) data: ArrayData,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ArrayData {
    I32Array(Vec<i32>),
    U32Array(Vec<u32>),
    F32Array(Vec<f32>),
    F64Array(Vec<f64>),
    U8Array(Vec<u8>),
    CompressedU8Array(Vec<u8>),
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PeerMessage {
    pub(crate) peer_id: u64,
    pub(crate) message: String,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Heartbeat {
    pub(crate) id: u64,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum MessageData {
    Init(Init),
    BinaryFileTransfer(BinaryFileTransfer),
    TextFileTransfer(TextFileTransfer),
    PollWorkerStatus(PollWorkerStatus),
    WorkerStatus(WorkerStatus),
    FileTransferRequest(FileTransferRequest),
    ArrayTransfer(ArrayTransfer),
    Heartbeat(Heartbeat),
    Peer(PeerMessage),
    Shutdown,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub id: u64,
    pub data: MessageData,
}
