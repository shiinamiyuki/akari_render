use akari::{os_pipe, scenegraph::node::*, serde_json};
use serde::{Deserialize, Serialize};
use std::{
    env::args,
    fs,
    io::Read,
    path::PathBuf,
    process::{exit, Command},
    time::{Duration, SystemTime},
};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct FileCmp {
    mtime: Duration,
    len: u64,
}
fn usage() {
    println!(r"usage(send): akari-sync -s <file> username@host:<file> <remote akari-sync-path>");
    println!(r"usage(recv): akari-sync -r");
}
fn send(local: &str, remote: (&str, &str), remote_sync: &str) -> bool {
    let mut ssh = Command::new("ssh");
    ssh.arg(remote.0);
    ssh.arg(remote_sync);
    ssh.arg("-r");
    ssh.arg(remote.1);
    let (mut reader, writer) = os_pipe::pipe().unwrap();
    ssh.stdout(writer.try_clone().unwrap());
    let mut handle = ssh.spawn().unwrap();
    std::mem::drop(ssh);

    let mut output = String::new();
    reader.read_to_string(&mut output).unwrap();
    if !handle.wait().unwrap().success() {
        return false;
    }
    let remote_f = serde_json::from_str::<FileCmp>(&output).unwrap();
    let local_f = get_metadata(local);
    if let Some(local_f) = local_f {
        if remote_f == local_f {
            println!("file no change");
            return true;
        }
    }
    let mut scp = Command::new("scp");
    scp.arg("-p");
    scp.arg(local);
    scp.arg(format!("{}:{}", remote.0, remote.1));
    scp.spawn().unwrap().wait().unwrap().success()
}
fn get_metadata(file: &str) -> Option<FileCmp> {
    if !PathBuf::from(file).exists() {
        return None;
    }
    let metadata = fs::metadata(file).expect(&format!("err reading metadata of {}", file));
    let mtime = metadata.modified().unwrap();
    let mtime = mtime.duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let len = metadata.len();
    Some(FileCmp { mtime, len })
}
fn main() {
    let args: Vec<_> = args().collect();
    let mode = &args[1];
    match mode.as_str() {
        "-s" => {
            let file = &args[2];
            let target = &args[3];
            let remote_sync = &args[4];
            let target: Vec<_> = target.split(":").collect();
            if !send(&file, (&target[0], &target[1]), &remote_sync) {
                eprintln!("failed sent {} to {}", args[2], args[3]);
                exit(-1);
            }
        }
        "-r" => {
            let file = &args[2];
            let cmp = get_metadata(file);
            let s = serde_json::to_string(&cmp).unwrap();
            print!("{}", s);
        }
        _ => {
            eprintln!("unrecognized mode {}", mode);
            usage();
            exit(-1);
        }
    }
}
