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

/// Hash functions in an attempt to match the behavior of Blender's Cycles.
pub mod blender {
    use super::*;
    #[tracked(crate = "luisa")]
    fn uint_to_float_excl(n: Expr<u32>) -> Expr<f32> {
        n.as_f32() * (1.0f32 / 4294967296.0)
    }
    #[tracked(crate = "luisa")]
    fn rot(x: impl AsExpr<Value = u32>, k: impl AsExpr<Value = u32>) -> Expr<u32> {
        let x = x.as_expr();
        let k = k.as_expr();
        (x << k) | (x >> (32 - k))
    }
    #[tracked(crate = "luisa")]
    fn mix(a: Var<u32>, b: Var<u32>, c: Var<u32>) {
        *a -= c;
        *a ^= rot(c, 4);
        *c += b;
        *b -= a;
        *b ^= rot(a, 6);
        *a += c;
        *c -= b;
        *c ^= rot(b, 8);
        *b += a;
        *a -= c;
        *a ^= rot(c, 16);
        *c += b;
        *b -= a;
        *b ^= rot(a, 19);
        *a += c;
        *c -= b;
        *c ^= rot(b, 4);
        *b += a;
    }
    #[tracked(crate = "luisa")]
    pub fn final_(a: Var<u32>, b: Var<u32>, c: Var<u32>) {
        *c ^= b;
        *c -= rot(b, 14);
        *a ^= c;
        *a -= rot(c, 11);
        *b ^= a;
        *b -= rot(a, 25);
        *c ^= b;
        *c -= rot(b, 16);
        *a ^= c;
        *a -= rot(c, 4);
        *b ^= a;
        *b -= rot(a, 14);
        *c ^= b;
        *c -= rot(b, 24);
    }
    #[tracked(crate = "luisa")]
    pub fn hash_uint(kx: Expr<u32>) -> Expr<u32> {
        let init: u32 = 0xdeadbeefu32 + (1u32 << 2) + 13u32;
        let a = init.var();
        let b = init.var();
        let c = init.var();
        outline(|| {
            *a += kx;
            final_(a, b, c);
        });
        c.load()
    }
    #[tracked(crate = "luisa")]
    pub fn hash_uint2(k: Expr<Uint2>) -> Expr<u32> {
        let init: u32 = 0xdeadbeefu32 + (2u32 << 2) + 13u32;
        let a = init.var();
        let b = init.var();
        let c = init.var();
        outline(|| {
            *a += k.y;
            *b += k.x;
            final_(a, b, c);
        });
        c.load()
    }
    #[tracked(crate = "luisa")]
    pub fn hash_uint3(k: Expr<Uint3>) -> Expr<u32> {
        let init: u32 = 0xdeadbeefu32 + (3u32 << 2) + 13u32;
        let a = init.var();
        let b = init.var();
        let c = init.var();
        outline(|| {
            *a += k.x;
            *b += k.y;
            *c += k.z;
            final_(a, b, c);
        });
        c.load()
    }
    #[tracked(crate = "luisa")]
    pub fn hash_uint4(k: Expr<Uint4>) -> Expr<u32> {
        let init: u32 = 0xdeadbeefu32 + (4u32 << 2) + 13u32;
        let a = init.var();
        let b = init.var();
        let c = init.var();
        outline(|| {
            *a += k.x;
            *b += k.y;
            *c += k.z;
            mix(a, b, c);
            *a += k.w;
            final_(a, b, c);
        });
        c.load()
    }
}
