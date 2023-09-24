use lazy_static::lazy_static;
// see https://github.com/LuisaGroup/LuisaRender/blob/next/src/util/rng.cpp
use crate::*;
use luisa::runtime::Callable;
lazy_static! {
    static ref XXHASH_32_1: Callable<fn(Expr<u32>) -> Expr<u32>> =
        Callable::<fn(Expr<u32>) -> Expr<u32>>::new_static(track!(|p: Expr<u32>| {
            const PRIME32_2: u32 = 2246822519;
            const PRIME32_3: u32 = 3266489917;
            const PRIME32_4: u32 = 668265263;
            const PRIME32_5: u32 = 374761393;
            let mut h32 = p + PRIME32_5;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = PRIME32_2 * (h32 ^ (h32 >> 15u32));
            h32 = PRIME32_3 * (h32 ^ (h32 >> 13u32));
            h32 ^ (h32 >> 16u32)
        }));
    static ref XXHASH_32_2: Callable<fn(Expr<Uint2>) -> Expr<u32>> =
        Callable::<fn(Expr<Uint2>) -> Expr<u32>>::new_static(track!(|p: Expr<Uint2>| {
            const PRIME32_2: u32 = 2246822519;
            const PRIME32_3: u32 = 3266489917;
            const PRIME32_4: u32 = 668265263;
            const PRIME32_5: u32 = 374761393;
            let mut h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = PRIME32_2 * (h32 ^ (h32 >> 15u32));
            h32 = PRIME32_3 * (h32 ^ (h32 >> 13u32));
            h32 ^ (h32 >> 16u32)
        }));
    static ref XXHASH_32_3: Callable<fn(Expr<Uint3>) -> Expr<u32>> =
        Callable::<fn(Expr<Uint3>) -> Expr<u32>>::new_static(track!(|p: Expr<Uint3>| {
            const PRIME32_2: u32 = 2246822519;
            const PRIME32_3: u32 = 3266489917;
            const PRIME32_4: u32 = 668265263;
            const PRIME32_5: u32 = 374761393;
            let mut h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = h32 + p.y * PRIME32_3;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = PRIME32_2 * (h32 ^ (h32 >> 15u32));
            h32 = PRIME32_3 * (h32 ^ (h32 >> 13u32));
            h32 ^ (h32 >> 16u32)
        }));
    static ref XXHASH_32_4: Callable<fn(Expr<Uint4>) -> Expr<u32>> =
        Callable::<fn(Expr<Uint4>) -> Expr<u32>>::new_static(track!(|p: Expr<Uint4>| {
            const PRIME32_2: u32 = 2246822519;
            const PRIME32_3: u32 = 3266489917;
            const PRIME32_4: u32 = 668265263;
            const PRIME32_5: u32 = 374761393;
            let mut h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = h32 + p.y * PRIME32_3;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = h32 + p.z * PRIME32_3;
            h32 = PRIME32_4 * ((h32 << 17u32) | (h32 >> (32u32 - 17u32)));
            h32 = PRIME32_2 * (h32 ^ (h32 >> 15u32));
            h32 = PRIME32_3 * (h32 ^ (h32 >> 13u32));
            h32 ^ (h32 >> 16u32)
        }));
}
pub fn xxhash32_1(p: Expr<u32>) -> Expr<u32> {
    XXHASH_32_1.call(p)
}
pub fn xxhash32_2(p: Expr<Uint2>) -> Expr<u32> {
    XXHASH_32_2.call(p)
}
pub fn xxhash32_3(p: Expr<Uint3>) -> Expr<u32> {
    XXHASH_32_3.call(p)
}
pub fn xxhash32_4(p: Expr<Uint4>) -> Expr<u32> {
    XXHASH_32_4.call(p)
}
