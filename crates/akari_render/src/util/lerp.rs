use crate::*;
#[tracked(crate = "luisa")]
pub fn nd_lerp(dim: usize, corners: &[Expr<f32>], pos: &[Expr<f32>]) -> Expr<f32> {
    assert_eq!(corners.len(), 1 << dim);
    assert_eq!(pos.len(), dim);
    let ret = 0.0f32.var();
    outline(|| {
        fn nd_lerp_impl(dim: usize, corners: &[Expr<f32>], pos: &[Expr<f32>]) -> Expr<f32> {
            if dim == 1 {
                corners[0].lerp(corners[1], pos[0])
            } else {
                let half = 1usize << (dim - 1);
                let left = &corners[..half];
                let right = &corners[half..];
                let left = nd_lerp_impl(dim - 1, left, &pos[..dim - 1]);
                let right = nd_lerp_impl(dim - 1, right, &pos[..dim - 1]);
                left.lerp(right, pos[dim - 1])
            }
        }
        *ret = nd_lerp_impl(dim, corners, pos);
    });
    ret.load()
}
