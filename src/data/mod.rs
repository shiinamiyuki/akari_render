pub struct GgxLtcfit {
    pub mat: [[f32; 9]; 64 * 64],
    pub amp: [f32; 64 * 64],
}
mod cie;
mod ltc;
mod sobolmat;
mod prime;
pub mod ior;
pub mod rgb2spec;
pub mod rgb8;
pub use cie::*;
pub use ltc::GGX_LTC_FIT;
pub use sobolmat::*;
pub use rgb8::*;
pub use ior::*;
pub use prime::*;