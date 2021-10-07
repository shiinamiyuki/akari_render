use std::{io::Read, path::PathBuf};

use crate::*;

pub struct CurrentDirGuard {
    current_dir: PathBuf,
}
impl CurrentDirGuard {
    pub fn new() -> Self {
        Self {
            current_dir: std::env::current_dir().unwrap(),
        }
    }
}
impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.current_dir).unwrap();
    }
}
pub fn serialize_pod<T: bytemuck::Pod>(pod: T, bytes: &mut Vec<u8>) {
    bytes.extend(bytemuck::cast_slice::<T, u8>(&[pod]));
}
pub fn serialize_pods<T: bytemuck::Pod>(pods: &Vec<T>, bytes: &mut Vec<u8>) {
    serialize_pod(pods.len(), bytes);
    bytes.extend_from_slice(bytemuck::cast_slice::<T, u8>(&pods[..]));
}

pub fn deserialize_pod<T: bytemuck::Pod, In: Read>(pod: &mut T, s: &mut In) {
    let tmp = unsafe { std::slice::from_raw_parts_mut(pod as *mut T, 1) };
    s.read(bytemuck::cast_slice_mut::<T, u8>(tmp)).unwrap();
}
pub fn deserialize_pods<T: bytemuck::Pod, In: Read>(pods: &mut Vec<T>, s: &mut In) {
    let mut len: usize = 0;
    deserialize_pod(&mut len, s);
    assert!(pods.is_empty());
    pods.reserve(len);
    {
        let slice = unsafe { std::slice::from_raw_parts_mut(pods.as_mut_ptr(), len) };
        s.read(bytemuck::cast_slice_mut::<T, u8>(slice)).unwrap();
    }
}
