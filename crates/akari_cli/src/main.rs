use std::io::{self, Write};

fn main() {
    // use akari_core::util::image::Image;
    // let img = Image::read("torus.png", akari_core::util::image::PixelFormat::RGBA8);
    // img.write("torus_copy.exr");
    use std::process::Command;

    let py = Command::new("python").args(&["-v"]).output().unwrap();
    io::stdout().write_all(&py.stdout).unwrap();
    io::stderr().write_all(&py.stderr).unwrap();
}
