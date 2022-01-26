pub struct GgxLtcfit {
    pub mat: [[f32; 9]; 64 * 64],
    pub amp: [f32; 64 * 64],
}
mod ltc;
mod sobolmat;
pub use ltc::GGX_LTC_FIT;
pub use sobolmat::*;