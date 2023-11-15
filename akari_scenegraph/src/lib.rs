use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    mem::MaybeUninit,
    ops::{Deref, Index},
    path::Path,
};

use serde::{de::Visitor, ser::SerializeMap, Deserialize, Serialize};

pub mod blender_util;
pub mod scene;
pub mod shader;

pub use scene::*;
pub use shader::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHeader {
    pub magic: [u8; 16],
    pub version: u32,
    pub length: u64,
}
impl BufferHeader {
    pub const MAGIC: [u8; 16] = *b"akari_scenegraph";
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeRef<T> {
    pub id: String,
    #[serde(skip)]
    phantom: std::marker::PhantomData<T>,
}
impl<T> PartialEq for NodeRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for NodeRef<T> {}
impl<T> PartialOrd for NodeRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}
impl<T> Ord for NodeRef<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl<T> std::hash::Hash for NodeRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}
impl<T> Deref for NodeRef<T> {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

#[derive(Clone, Debug)]
pub struct Collection<T>(BTreeMap<NodeRef<T>, T>);
impl<'a, T> IntoIterator for &'a Collection<T> {
    type Item = (&'a NodeRef<T>, &'a T);
    type IntoIter = std::collections::btree_map::Iter<'a, NodeRef<T>, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<T> std::ops::Deref for Collection<T> {
    type Target = BTreeMap<NodeRef<T>, T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> std::ops::DerefMut for Collection<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T> Serialize for Collection<T>
where
    T: Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, value) in self.iter() {
            map.serialize_entry(&key.id, value)?;
        }
        map.end()
    }
}
struct CollectionVisitor<T> {
    phantom: std::marker::PhantomData<T>,
}
impl<'de, T: Deserialize<'de>> Visitor<'de> for CollectionVisitor<T> {
    type Value = Collection<T>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a map with string keys and values of type T")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut collection = Collection(BTreeMap::new());
        while let Some((key, value)) = map.next_entry::<String, T>()? {
            collection.insert(
                NodeRef {
                    id: key,
                    phantom: std::marker::PhantomData,
                },
                value,
            );
        }
        Ok(collection)
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Collection<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(CollectionVisitor {
            phantom: std::marker::PhantomData,
        })
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ExtStridedSlice {
    ptr: u64,
    len: u64,
    stride: u64,
}
impl ExtStridedSlice {
    #[inline]
    pub fn new(ptr: u64, len: u64, stride: u64) -> Self {
        Self { ptr, len, stride }
    }
    #[inline]
    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }
    #[inline]
    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub unsafe fn get<T>(&self, index: usize) -> *const T {
        debug_assert!(index < self.len());
        self.as_ptr::<T>().add(index * self.stride())
    }
    #[inline]
    pub unsafe fn get_mut<T>(&self, index: usize) -> *mut T {
        debug_assert!(index < self.len());
        self.as_mut_ptr::<T>().add(index * self.stride())
    }
}
pub fn write_binary<P: AsRef<Path>, T>(path: P, data: &[T]) -> std::io::Result<()>
where
    T: Copy + Sized,
{
    let mut file = File::create(path)?;
    let header = BufferHeader {
        magic: BufferHeader::MAGIC,
        version: VERSION,
        length: std::mem::size_of::<T>() as u64 * data.len() as u64,
    };
    unsafe {
        file.write_all(std::slice::from_raw_parts(
            &header as *const BufferHeader as *const u8,
            std::mem::size_of::<BufferHeader>(),
        ))?;
        file.write_all(std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            std::mem::size_of::<T>() * data.len(),
        ))?;
    }
    Ok(())
}

pub fn read_binary<P: AsRef<Path>, T>(path: P) -> std::io::Result<Vec<T>>
where
    T: Copy + Sized,
{
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut header = MaybeUninit::<BufferHeader>::uninit();
    unsafe {
        file.read_exact(std::slice::from_raw_parts_mut(
            header.as_mut_ptr() as *mut u8,
            std::mem::size_of::<BufferHeader>(),
        ))?;
        let header = header.assume_init();
        if header.magic != BufferHeader::MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic",
            ));
        }
        if header.version != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid version",
            ));
        }
        if header.length % std::mem::size_of::<T>() as u64 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid length",
            ));
        }
        let cnt = header.length as usize / std::mem::size_of::<T>();
        let mut data = vec![MaybeUninit::<T>::uninit(); cnt];
        file.read_exact(std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            std::mem::size_of::<T>() * cnt,
        ))?;
        Ok(std::mem::transmute::<_, Vec<T>>(data))
    }
}

pub const VERSION: u32 = 1000;
