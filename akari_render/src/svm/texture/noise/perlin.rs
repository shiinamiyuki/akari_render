use crate::{
    util::{hash::blender, lerp::nd_lerp},
    *,
};

#[tracked]
fn fade(t: Expr<f32>) -> Expr<f32> {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
#[tracked]
fn negate_if(val: Expr<f32>, condition: Expr<bool>) -> Expr<f32> {
    select(condition, -val, val)
}

#[tracked]
fn grad1(hash: Expr<u32>, x: Expr<f32>) -> Expr<f32> {
    let h = hash & 15;
    let grad = 1.0 + (h & 7).cast_f32();
    negate_if(grad, (h & 8) != 0) * x
}
#[tracked]
pub fn grad2(hash: Expr<u32>, x: Expr<f32>, y: Expr<f32>) -> Expr<f32> {
    let h = hash & 7;
    let u = if h < 4 { x } else { y };
    let v = 2.0f32 * (if h < 4 { y } else { x });
    negate_if(u, (h & 1) != 0) + negate_if(v, (h & 2) != 0)
}
#[tracked]
fn perlin_1d(x: Expr<f32>) -> Expr<f32> {
    let ix = x.floor().cast_i32();
    let fx = x - ix.cast_f32();
    let u = fade(fx);
    grad1(blender::hash_uint(ix.as_u32()), fx)
        .lerp(grad1(blender::hash_uint(ix.as_u32() + 1), fx - 1.0), u)
}
#[tracked]
fn perlin_2d(x: Expr<Float2>) -> Expr<f32> {
    let y = x.y;
    let x = x.x;

    let ix = x.floor().cast_i32();
    let iy = y.floor().cast_i32();
    let fx = x - ix.cast_f32();
    let fy = y - iy.cast_f32();

    let u = fade(fx);
    let v = fade(fy);

    let r = nd_lerp(
        2,
        &(0..4)
            .map(|i| {
                let ix_offset = (i & 1) as i32;
                let iy_offset = (i >> 1) as i32;
                grad2(
                    blender::hash_uint2(Int2::expr(ix + ix_offset, iy + iy_offset).cast_u32()),
                    fx - ix_offset as f32,
                    fy - iy_offset as f32,
                )
            })
            .collect::<Vec<_>>(),
        &[u, v],
    );
    r
}
