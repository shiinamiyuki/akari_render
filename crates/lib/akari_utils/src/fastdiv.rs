use akari_common::glam::{uvec2, UVec2};

use crate::log2;

#[derive(Clone, Copy, Debug)]
pub struct FastDiv32 {
    shift: u64,
    add: u64,
    mul: u64,
}

//https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers
impl FastDiv32 {
    const N: u64 = 32;
    pub fn new(d: u32) -> Self {
        let mut div = Self {
            shift: 0,
            add: 0,
            mul: 0,
        };
        let l = (d as f64).log2().floor() as u64;
        if d.is_power_of_two() {
            div.mul = u32::MAX as u64;
            div.add = u32::MAX as u64;
        } else {
            let m_down = ((1u64) << (Self::N + l)) / d as u64;
            let m_up = m_down + 1;
            let temp = m_up * d as u64;
            let use_round_up_method = temp <= (1 << l);
            if use_round_up_method {
                div.mul = m_up;
                div.add = 0;
            } else {
                div.mul = m_down;
                div.add = m_down;
            }
        }
        div.shift = l;
        div
    }
}
impl std::ops::Div<FastDiv32> for u32 {
    type Output = u32;

    fn div(self, rhs: FastDiv32) -> Self::Output {
        let full_product = (self as u64) * rhs.mul + rhs.add;
        ((full_product >> FastDiv32::N) >> rhs.shift) as u32
    }
}

impl std::ops::Div<FastDiv32> for UVec2 {
    type Output = UVec2;

    fn div(self, rhs: FastDiv32) -> Self::Output {
        uvec2(self.x / rhs, self.y / rhs)
    }
}

mod test {
    #[test]
    fn test_div() {
        use super::*;
        for i in 0..16 * 1024 {
            for d in 2..1024 {
                let div = FastDiv32::new(d);
                assert_eq!(
                    i / d,
                    i / div,
                    "fastdiv failed for i={}, d={}, data={:?}",
                    i,
                    d,
                    div
                );
            }
        }
    }
}
