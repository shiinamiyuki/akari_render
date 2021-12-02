use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub struct Server {
    listener: TcpListener,
    connections: Vec<Arc<Connection>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Worker,
    Client,
}
struct Connection {
    socket: Mutex<TcpStream>,
    role: Role,
}
impl Server {
    // pub fn alive_workers(&self)->usize{

    // }
    //     pub fn handle_client(){}
    //     pub async fn main_loop(&mut self)->std::io::Result<()>{
    //         loop{
    //             let (socket, _) = self.listener.accept().await?;
    //             let socket = Mutex::new(socket);
    //             let client = Arc::new(ClientConnection{
    //                 socket
    //             });
    //             self.clients.push(client.clone());
    //             tokio::spawn(async move {

    //             });
    //         }
    //     }
}
