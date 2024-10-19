fn main() {
    use akari_core::util::image::Image;
    let img = Image::read("torus.png", akari_core::util::image::PixelFormat::RGBA8);
    img.write("torus_copy.png");
}
