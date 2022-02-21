use std::{
    io::{Read, Write},
    mem::MaybeUninit,
};

/* minimal binary ser/de
 * we cannot use bincode 2.0 as it is unstable as of 01-31-2022
 * we dont want to use bson because we want faster read/write
 * 
 */


pub trait Decode {
    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized;
}
pub trait Encode {
    fn encode<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
}
#[macro_export]
macro_rules! impl_binserde {
    ($t:ty) => {
        impl Decode for $t {
            fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self>
            where
                Self: Sized,
            {
                let mut data = std::mem::MaybeUninit::<$t>::uninit();
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        data.as_mut_ptr() as *mut u8,
                        std::mem::size_of::<$t>(),
                    )
                };
                reader.read_exact(slice)?;
                unsafe { Ok(data.assume_init()) }
            }
        }
        impl Encode for $t {
            fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        self as *const Self as *const u8,
                        std::mem::size_of::<$t>(),
                    )
                };
                writer.write_all(slice)?;
                Ok(())
            }
        }
        impl Decode for Vec<$t> {
            fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self>
            where
                Self: Sized,
            {
                let stride = std::mem::size_of::<$t>();
                let mut len: [u8; 8] = [0; 8];
                reader.read_exact(&mut len)?;
                let len = u64::from_le_bytes(len) as usize;
                let mut data = Vec::<$t>::with_capacity(len);
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len * stride)
                };
                reader.read_exact(slice)?;
                unsafe { data.set_len(len) }
                Ok(data)
            }
        }
        impl Encode for [$t] {
            fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                let len: [u8; 8] = (self.len() as u64).to_le_bytes();
                let stride = std::mem::size_of::<$t>();
                let slice = unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * stride)
                };
                writer.write_all(&len)?;
                writer.write_all(slice)?;
                Ok(())
            }
        }
    };
}
impl_binserde!(f32);
impl_binserde!([f32; 2]);
impl_binserde!([f32; 3]);
impl_binserde!(u32);
impl_binserde!([u32; 2]);
impl_binserde!([u32; 3]);
impl_binserde!(u8);

impl Encode for String {
    fn encode<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.as_bytes().encode(writer)
    }
}
impl Decode for String {
    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let v = Vec::<u8>::decode(reader)?;
        Ok(Self::from_utf8_lossy(&v).into_owned())
    }
}

mod test {

    #[test]
    fn test_serde() {
        use super::*;
        use std::io::Cursor;
        let s0: String = "Hello world".into();
        let s1: String = "Binary serde good!".into();
        let v0: Vec<[f32; 3]> = vec![[0.2, 0.3, 0.4]; 16];
        let v1: Vec<u32> = (0..128).collect();
        let mut buf = Cursor::new(vec![0u8; 10240]);
        s0.encode(&mut buf).unwrap();
        s1.encode(&mut buf).unwrap();
        v0.encode(&mut buf).unwrap();
        v1.encode(&mut buf).unwrap();
        // buf.set_position(0);
        let buf = buf.into_inner();
        let mut slice = buf.as_slice();
        let s02: String = Decode::decode::<&[u8]>(&mut slice).unwrap();
        let s12: String = Decode::decode::<&[u8]>(&mut slice).unwrap();
        let v02: Vec<[f32; 3]> = Decode::decode::<&[u8]>(&mut slice).unwrap();
        let v12: Vec<u32> = Decode::decode::<&[u8]>(&mut slice).unwrap();
        assert_eq!(s02, s0);
        assert_eq!(s12, s1);
        assert_eq!(v02, v0);
        assert_eq!(v12, v1);
    }
}
