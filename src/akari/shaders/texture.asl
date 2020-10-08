struct TextureHandle {
    int type;
    int index;
}
struct FloatTexture {
    float value;
}
const int FloatTextureType = 1;
struct RGBTexture {
    vec3 value;
}
const int RGBTextureType = 2;

buffer FloatTexture[] float_textures;
buffer RGBTexture[] rgb_textures;
