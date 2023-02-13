use crate::{
    light::Light,
    surface::Surface,
    texture::{ColorTexture, FloatTexture},
    *,
};

pub struct Scene {
    pub float_textures: Polymorphic<dyn FloatTexture>,
    pub color_textures: Polymorphic<dyn ColorTexture>,
    pub surfaces: Polymorphic<dyn Surface>,
    pub lights: Polymorphic<dyn Light>,
}
