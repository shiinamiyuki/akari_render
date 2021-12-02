use bson::{doc, Document};
use serde::{Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
pub async fn send_message(stream: &mut TcpStream, bson: Document) -> std::io::Result<()> {
    let mut buf: Vec<u8> = vec![];
    bson.to_writer(&mut buf).unwrap();
    let len = (buf.len() as u64).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&buf).await?;
    Ok(())
}

pub async fn send_message2<T: Serialize>(
    stream: &mut TcpStream,
    ty: String,
    data: &T,
) -> std::io::Result<()> {
    let bson = doc! {
        "type":ty,
        "content":bson::to_bson(&data).unwrap()
    };
    send_message(stream, bson).await
}
pub async fn recv_message(stream: &mut TcpStream) -> bson::ser::Result<Document> {
    let mut len = [0u8; std::mem::size_of::<u64>()];
    stream.read_exact(&mut len).await?;
    let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
    stream.read_exact(&mut buf).await?;
    Ok(Document::from_reader(&mut buf.as_slice()).unwrap())
}
