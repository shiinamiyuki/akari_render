use crate::{color::ColorRepr, interaction::SurfaceInteraction, *};

pub trait Bsdf {}
pub trait Surface {
    fn closure(&self, si: Expr<SurfaceInteraction>, color_repr: &ColorRepr);
}
