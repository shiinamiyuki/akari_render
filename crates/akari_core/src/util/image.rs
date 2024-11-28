pub use akari_cpp_ext::akari_image_PixelFormat as PixelFormat;

pub struct Image {
    api: akari_cpp_ext::akari_image_ImageApi,
    inner: akari_cpp_ext::akari_image_Image,
}
impl Image {
    #[inline]
    pub fn width(&self) -> u32 {
        self.inner.width as u32
    }
    #[inline]
    pub fn height(&self) -> u32 {
        self.inner.height as u32
    }
    #[inline]
    pub fn format(&self) -> PixelFormat {
        self.inner.format
    }
    #[inline]
    pub fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.inner.data, self.width() as usize * self.height() as usize * self.format().size()) }
    }
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.inner.data, self.width() as usize * self.height() as usize * self.format().size()) }
    }
    pub fn read(path: impl AsRef<str>, format: PixelFormat) -> Self {
        let path = std::ffi::CString::new(path.as_ref()).unwrap();
        let api = unsafe { akari_cpp_ext::extension().create_image_api() };
        let inner = unsafe { (api.read.unwrap())(path.as_ptr(), format) };
        Self {
            api,
            inner,
        }
    }
    pub fn write(&self, path: impl AsRef<str>) -> bool {
        let path = std::ffi::CString::new(path.as_ref()).unwrap();
        unsafe { (self.api.write.unwrap())(path.as_ptr(), &self.inner) }
    }
}
impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            (self.api.destroy_image.unwrap())(&self.inner);
        }
    }
}
