use std::ffi::c_char;

#[link(name = "akari_cpp_ext")]
extern "C" {
    //void rgb2spec_opt(int argc, char **argv);
    pub fn rgb2spec_opt(argc: i32, argv: *const *const c_char) -> i32;
}
