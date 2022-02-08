use std::ops::Index;

use crate::half::f16;
use crate::{
    arrayvec::VirtualStorage,
    binserde::{Decode, Encode},
    fastdiv::FastDiv32,
    log2, srgb_to_linear, srgb_to_linear1, RobustSum,
};
use crate::{linear_to_srgb, linear_to_srgb1, srgb_to_linear1_u8, srgb_to_linear_u8};
use akari_common::glam::{uvec2, vec3, vec4, IVec2, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};

use super::arrayvec::{ArrayVec, DynStorage};
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum PixelFormat {
    R8 = 0,  // linear space
    SR8,     // srgb space
    Rgb8,    // linear space
    SRgba8,  // srgb space, u8
    Rgba8,   // linear space, u8
    SRgb8,   // srgb space, u8
    Rgb16f,  // linear space, fp16
    Rgba16f, // linear space, fp16
    Rgb32f,  // linear space, fp32
    Rgba32f, // linear space, fp32
}
impl PixelFormat {
    pub const fn formats() -> [PixelFormat; 10] {
        [
            PixelFormat::R8,
            PixelFormat::SR8,
            PixelFormat::Rgb8,
            PixelFormat::SRgba8,
            PixelFormat::SRgb8,
            PixelFormat::Rgb16f,
            PixelFormat::Rgb32f,
            PixelFormat::Rgba16f,
            PixelFormat::Rgba32f,
            PixelFormat::Rgba8,
        ]
    }
    #[inline(always)]
    pub const fn size(self) -> usize {
        match self {
            PixelFormat::R8 | PixelFormat::SR8 => 1,
            PixelFormat::Rgb8 | PixelFormat::SRgb8 => 3,
            PixelFormat::Rgba8 | PixelFormat::SRgba8 => 4,
            PixelFormat::Rgb16f => 6,
            PixelFormat::Rgba16f => 8,
            PixelFormat::Rgb32f => 12,
            PixelFormat::Rgba32f => 16,
        }
    }
    #[inline(always)]
    pub const fn num_channels(self) -> usize {
        match self {
            PixelFormat::R8 => 1,
            PixelFormat::SR8 => 1,
            PixelFormat::Rgb8 => 3,
            PixelFormat::SRgba8 => 4,
            PixelFormat::SRgb8 => 3,
            PixelFormat::Rgb16f => 3,
            PixelFormat::Rgb32f => 3,
            PixelFormat::Rgba16f => 4,
            PixelFormat::Rgba32f => 4,
            PixelFormat::Rgba8 => 4,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageMetadata {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub format: PixelFormat,
}
pub const TILE_SIZE_BTYES: usize = 12 * 1024;
impl_binserde!(ImageMetadata);

#[repr(align(4096))]
#[derive(Clone, Copy)]
pub struct RawImageTile([u8; TILE_SIZE_BTYES]);

impl_binserde!(RawImageTile);
#[derive(Clone)]
pub struct TiledImage {
    metadata: ImageMetadata,
    div_tile_size: FastDiv32,
    ntiles: UVec2,
    data: Vec<RawImageTile>,
}

impl Encode for TiledImage {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.metadata.encode(writer)?;
        self.data.encode(writer)
    }
}

impl Decode for TiledImage {
    fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let metadata: ImageMetadata = Decode::decode(reader)?;
        let data = Decode::decode(reader)?;
        Ok(Self {
            metadata,
            data,
            div_tile_size: FastDiv32::new(metadata.tile_size),
            ntiles: (uvec2(metadata.width, metadata.height) + metadata.tile_size - 1)
                / metadata.tile_size,
        })
    }
}
fn load4(bytes_: &[u8], format: PixelFormat) -> Vec4 {
    let mut bytes = [0u8; 16];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes_.as_ptr(), bytes.as_mut_ptr(), format.size());
    }
    match format {
        PixelFormat::R8 => Vec3::splat(bytes[0] as f32 / 255.0).extend(1.0),
        PixelFormat::SR8 => Vec3::splat(srgb_to_linear1_u8(bytes[0])).extend(1.0),
        PixelFormat::Rgb8 => vec3(
            bytes[0] as f32 / 255.0,
            bytes[1] as f32 / 255.0,
            bytes[2] as f32 / 255.0,
        )
        .extend(1.0),
        PixelFormat::SRgb8 => srgb_to_linear_u8([bytes[0], bytes[1], bytes[2]]).extend(1.0),
        PixelFormat::Rgba8 => vec4(
            bytes[0] as f32 / 255.0,
            bytes[1] as f32 / 255.0,
            bytes[2] as f32 / 255.0,
            bytes[3] as f32 / 255.0,
        ),
        PixelFormat::SRgba8 => {
            srgb_to_linear_u8([bytes[0], bytes[1], bytes[2]]).extend(bytes[3] as f32 / 255.0)
        }
        PixelFormat::Rgb16f => {
            let rgb16: [f16; 3] = [
                f16::from_le_bytes([bytes[0], bytes[1]]),
                f16::from_le_bytes([bytes[2], bytes[3]]),
                f16::from_le_bytes([bytes[4], bytes[5]]),
            ];
            vec4(rgb16[0].to_f32(), rgb16[1].to_f32(), rgb16[2].to_f32(), 1.0)
        }
        PixelFormat::Rgba16f => {
            let rgba16: [f16; 4] = [
                f16::from_le_bytes([bytes[0], bytes[1]]),
                f16::from_le_bytes([bytes[2], bytes[3]]),
                f16::from_le_bytes([bytes[4], bytes[5]]),
                f16::from_le_bytes([bytes[6], bytes[7]]),
            ];
            vec4(
                rgba16[0].to_f32(),
                rgba16[1].to_f32(),
                rgba16[2].to_f32(),
                rgba16[3].to_f32(),
            )
        }
        PixelFormat::Rgb32f => {
            let rgb32 = vec3(
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            );
            rgb32.extend(1.0)
        }
        PixelFormat::Rgba32f => {
            let rgba32 = vec4(
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
                f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            );
            rgba32
        }
    }
}
fn store4(bytes_: &mut [u8], value: Vec4, format: PixelFormat) {
    let mut bytes = [0u8; 16];
    match format {
        PixelFormat::R8 => {
            bytes[0] = (value.x * 255.0).clamp(0.0, 255.0) as u8;
        }
        PixelFormat::SR8 => {
            bytes[0] = (linear_to_srgb1(value.x.clamp(0.0, 1.0)) * 255.0).clamp(0.0, 255.0) as u8;
        }
        PixelFormat::Rgb8 => {
            let rgb = value.xyz();
            bytes[0] = (rgb.x * 255.0).clamp(0.0, 255.0) as u8;
            bytes[1] = (rgb.y * 255.0).clamp(0.0, 255.0) as u8;
            bytes[2] = (rgb.z * 255.0).clamp(0.0, 255.0) as u8;
        }
        PixelFormat::SRgba8 => {
            let rgb = linear_to_srgb(value.xyz().clamp(Vec3::ZERO, Vec3::ONE));
            bytes[0] = (rgb.x * 255.0).clamp(0.0, 255.0) as u8;
            bytes[1] = (rgb.y * 255.0).clamp(0.0, 255.0) as u8;
            bytes[2] = (rgb.z * 255.0).clamp(0.0, 255.0) as u8;
            bytes[3] = (value.w * 255.0).clamp(0.0, 255.0) as u8;
        }
        PixelFormat::Rgba8 => {
            bytes[0] = (value.x * 255.0).clamp(0.0, 255.0) as u8;
            bytes[1] = (value.y * 255.0).clamp(0.0, 255.0) as u8;
            bytes[2] = (value.z * 255.0).clamp(0.0, 255.0) as u8;
            bytes[3] = (value.w * 255.0).clamp(0.0, 255.0) as u8;
        }
        PixelFormat::SRgb8 => {
            let rgba = linear_to_srgb(value.xyz().clamp(Vec3::ZERO, Vec3::ONE)).extend(value.w);
            bytes[0] = (rgba.x * 255.0).clamp(0.0, 255.0) as u8;
            bytes[1] = (rgba.y * 255.0).clamp(0.0, 255.0) as u8;
            bytes[2] = (rgba.z * 255.0).clamp(0.0, 255.0) as u8;
            bytes[3] = (rgba.w * 255.0).clamp(0.0, 255.0) as u8;
        }
        PixelFormat::Rgb16f => {
            let rgb: [f16; 3] = [
                f16::from_f32(value.x),
                f16::from_f32(value.y),
                f16::from_f32(value.z),
            ];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rgb.as_ptr() as *const u8,
                    bytes_.as_mut_ptr(),
                    format.size(),
                );
                return;
            }
        }
        PixelFormat::Rgba16f => {
            let rgba: [f16; 4] = [
                f16::from_f32(value.x),
                f16::from_f32(value.y),
                f16::from_f32(value.z),
                f16::from_f32(value.w),
            ];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rgba.as_ptr() as *const u8,
                    bytes_.as_mut_ptr(),
                    format.size(),
                );
                return;
            }
        }
        PixelFormat::Rgb32f => unsafe {
            let value = value.to_array();
            std::ptr::copy_nonoverlapping(
                value.as_ptr() as *const u8,
                bytes_.as_mut_ptr(),
                format.size(),
            );
            return;
        },
        PixelFormat::Rgba32f => unsafe {
            let value = value.to_array();
            std::ptr::copy_nonoverlapping(
                value.as_ptr() as *const u8,
                bytes_.as_mut_ptr(),
                format.size(),
            );
            return;
        },
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), bytes_.as_mut_ptr(), format.size());
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WrappingMode {
    Zero,
    Clamp,
    Repeat,
}
impl TiledImage {
    pub fn metadata(&self) -> &ImageMetadata {
        &self.metadata
    }
    pub fn dimension(&self) -> UVec2 {
        uvec2(self.metadata.width, self.metadata.height)
    }
    fn index_offset(&self, p: UVec2) -> (usize, usize) {
        let p_tile = p / self.div_tile_size;
        let tile_idx = p_tile.x + p_tile.y * self.ntiles.x;
        let offset = p - p_tile * self.metadata.tile_size;
        let pixel_idx = offset.x + offset.y * self.metadata.tile_size;
        let stride = self.metadata.format.size();
        let i = pixel_idx as usize * stride;
        (tile_idx as usize, i)
    }
    pub fn store(&mut self, p: UVec2, value: Vec4) {
        let res = self.dimension();
        let oob = (p.cmplt(UVec2::ZERO) | p.cmpge(res)).any();
        assert!(!oob);
        let (tile_idx, i) = self.index_offset(p);
        let tile = &mut self.data[tile_idx];
        let stride = self.metadata.format.size();
        let bytes = &mut tile.0[i..i + stride];
        store4(bytes, value, self.metadata.format)
    }
    pub fn load(&self, mut p: IVec2, wrap: WrappingMode) -> Vec4 {
        let res = self.dimension().as_ivec2();

        match wrap {
            WrappingMode::Clamp => {
                p = p.clamp(IVec2::ZERO, res - 1);
            }
            WrappingMode::Repeat => {
                p = (p % res + res) % res;
            }
            WrappingMode::Zero => {
                let oob = (p.cmplt(IVec2::ZERO) | p.cmpge(res)).any();
                if oob {
                    return Vec4::ZERO;
                }
            }
        }
        debug_assert!((p.cmpge(IVec2::ZERO)).all(), "{:?} {:?}", p, wrap);
        let p = p.as_uvec2();
        let (tile_idx, i) = self.index_offset(p);
        debug_assert!(tile_idx < self.data.len(), "{:?}", p);
        let tile = &self.data[tile_idx];
        let stride = self.metadata.format.size();
        let bytes = &tile.0[i..i + stride];
        load4(bytes, self.metadata.format)
    }
    pub fn loadf(&self, p: Vec2, wrap: WrappingMode) -> Vec4 {
        let res = uvec2(self.metadata.width, self.metadata.height).as_ivec2();
        let ip = p * res.as_vec2();
        self.load(ip.as_ivec2(), wrap)
    }
    pub fn tile_size(format: PixelFormat) -> u32 {
        let npixel = TILE_SIZE_BTYES / format.size();
        let sz = (npixel as f64).sqrt().floor() as usize;
        assert!(sz * sz * format.size() <= TILE_SIZE_BTYES);
        sz as u32
    }
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let metadata = ImageMetadata {
            width,
            height,
            format,
            tile_size: Self::tile_size(format),
        };
        let ntiles =
            (uvec2(metadata.width, metadata.height) + metadata.tile_size - 1) / metadata.tile_size;
        Self {
            metadata,
            data: vec![RawImageTile([0; TILE_SIZE_BTYES]); (ntiles.x * ntiles.y) as usize],
            ntiles,
            div_tile_size: FastDiv32::new(metadata.tile_size),
        }
    }
    pub fn from_fn<F: Fn(u32, u32) -> Vec4>(
        width: u32,
        height: u32,
        format: PixelFormat,
        f: F,
    ) -> Self {
        let mut img = Self::new(width, height, format);
        img.fill_fn(f);
        img
    }
    pub fn fill_fn<F: Fn(u32, u32) -> Vec4>(&mut self, f: F) {
        for h in 0..self.metadata.height {
            for w in 0..self.metadata.width {
                self.store(uvec2(w, h), f(w, h));
            }
        }
    }
}

mod test {

    #[test]
    fn test_sl() {
        use super::*;
        use akari_common::rand::thread_rng;
        use akari_common::rand::Rng;

        let mut rng = thread_rng();
        for format in PixelFormat::formats() {
            let prec = match format {
                PixelFormat::R8
                | PixelFormat::SR8
                | PixelFormat::Rgb8
                | PixelFormat::Rgba8
                | PixelFormat::SRgb8
                | PixelFormat::SRgba8 => 0.01,
                _ => 0.001,
            };
            for _ in 0..10240 {
                let v = vec4(rng.gen(), rng.gen(), rng.gen(), rng.gen());
                let mut bytes = [0u8; 16];
                store4(&mut bytes, v, format);
                let u = load4(&bytes, format);
                for c in 0..format.num_channels() {
                    assert!(
                        (u[c] - v[c]).abs() < prec,
                        "format: {:?} chanel {} mismatch, values are:{} {}",
                        format,
                        c,
                        u[c],
                        v[c]
                    );
                }
            }
        }
    }
}
