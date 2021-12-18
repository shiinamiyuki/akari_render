use std::net::Shutdown;
use std::sync::{Arc, Mutex, RwLock};

use bson::Document;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use super::message::{recv_message, Message, MessageData, Role};

pub struct Server {
    listener: TcpListener,
    connections: Vec<Arc<Connection>>,
}

struct Connection {
    socket: Arc<Mutex<TcpStream>>,
    role: Role,
    alive: RwLock<bool>,
}

fn log_error<T: std::fmt::Debug, E: std::fmt::Debug>(
    r: &Result<T, E>,
    socket: &TcpStream,
    err_msg: &str,
) {
    if r.is_ok() {
        return;
    }
    if let Ok(addr) = socket.peer_addr() {
        log::error!("{}, remote: {:?} what: {:?}", err_msg, addr, r);
    } else if let Ok(addr) = socket.local_addr() {
        log::error!("{}, local: {:?}, what: {:?}", err_msg, addr, r);
    } else {
        log::error!("{}, unknown, what: {:?}", err_msg, r);
    }
}
fn log_error2(socket: &TcpStream, err_msg: &str) {
    if let Ok(addr) = socket.peer_addr() {
        log::error!("{}, remote: {:?}", err_msg, addr);
    } else if let Ok(addr) = socket.local_addr() {
        log::error!("{}, local: {:?}", err_msg, addr);
    } else {
        log::error!("{}, unknown", err_msg);
    }
}
// fn get_message_type<'a>(socket: &TcpStream, document: &'a Document) -> Option<&'a str> {
//     if let Some(msg_type) = document.get("type") {
//         if let Some(msg_type) = msg_type.as_str() {
//             Some(msg_type)
//         } else {
//             log_error2(&socket, "message does not have field 'type'");
//             None
//         }
//     } else {
//         log_error2(&socket, "message does not have field 'type'");
//         None
//     }
// }
impl Server {
    // pub fn alive_workers(&self)->usize{

    // }
    //     pub fn handle_client(){}
    async fn handle_client(connection: &Arc<Connection>) -> bool {
        loop {
            let mut socket = connection.socket.lock().unwrap();
            match recv_message(&mut socket).await {
                Ok(msg) => match &msg.data {
                    MessageData::Init(_) => {
                        log_error2(&socket, "cannot reinit client/worker");
                        return false;
                    }
                    MessageData::BinaryFileTransfer(_) => todo!(),
                    MessageData::TextFileTransfer(_) => todo!(),
                    MessageData::PollWorkerStatus(_) => todo!(),
                    MessageData::WorkerStatus(_) => todo!(),
                    MessageData::FileTransferRequest(_) => todo!(),
                    MessageData::ArrayTransfer(_) => todo!(),
                    MessageData::Heartbeat(_) => todo!(),
                    MessageData::Peer(_) => todo!(),
                    MessageData::Shutdown => {
                        return true;
                    }
                },
                Err(e) => {
                    let e = Err(e);
                    log_error::<Message, _>(&e, &socket, "recv message");
                    return false;
                }
            }
        }
    }
    async fn handle_incoming(&mut self) -> std::io::Result<()> {
        let (mut socket, _) = self.listener.accept().await?;

        let r = recv_message(&mut socket).await;
        if let Ok(init_msg) = r {
            match &init_msg.data {
                MessageData::Init(init) => {
                    let socket = Arc::new(Mutex::new(socket));
                    let connection = Arc::new(Connection {
                        role: init.role,
                        socket: socket.clone(),
                        alive: RwLock::new(true),
                    });
                    let lock = Arc::new(Mutex::new(()));
                    let g = lock.lock().unwrap();
                    {
                        let lock = lock.clone();
                        let connection = connection.clone();

                        tokio::spawn(async move {
                            let _g = lock.lock().unwrap();
                            Self::handle_client(&connection);
                            *connection.alive.write().unwrap() = false;
                        });
                    }
                    std::mem::drop(g);
                    self.connections.push(connection);
                }
                _ => {}
            }
        } else {
            log_error(&r, &socket, "fail to init");
        }
        Ok(())
    }

    pub async fn main_loop(&mut self) -> std::io::Result<()> {
        loop {
            self.handle_incoming().await?;
        }
    }
}
