macro_rules! def_vecn_by_concat {
    ($(#[$attr:meta])* struct $name:ident | $first:ty, $second:ty | $t:ty,$n:literal) => {
        $(#[$attr])*
        pub struct $name {
            first: $first,
            second: $second,
        }
        impl $name{
            pub const ZERO:Self = Self{first:<$first>::ZERO,second:<$second>::ZERO};
            pub const ONE:Self = Self{first:<$first>::ONE,second:<$second>::ONE};            
        }
        impl std::ops::Add for $name {
            type Output = Self;
            fn add(self, rhs:Self)->Self{
                Self{
                    first:self.first + rhs.first,
                    second:self.second + rhs.second
                }
            }
        }
    };
}

def_vecn_by_concat!(struct Vec8 | Vec4, Vec4 | f32, 8);
// fn test(){
//     let v:Vec8
// }

use akari_common::glam::Vec4;