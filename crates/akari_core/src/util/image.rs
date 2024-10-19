pub use akari_cpp_ext::akari_image_PixelFormat as PixelFormat;

pub struct Image {
    _api: akari_cpp_ext::akari_image_ImageApi,
    _inner: akari_cpp_ext::akari_image_Image,
}
impl Image {
    #[inline]
    pub fn width(&self) -> u32 {
        self._inner.width as u32
    }
    #[inline]
    pub fn height(&self) -> u32 {
        self._inner.height as u32
    }
    #[inline]
    pub fn format(&self) -> PixelFormat {
        self._inner.format
    }
    #[inline]
    pub fn data(&self) -> *mut u8 {
        self._inner.data
    }
    pub fn read(path: impl AsRef<str>, format: PixelFormat) -> Self {
        let path = std::ffi::CString::new(path.as_ref()).unwrap();
        let api = unsafe { akari_cpp_ext::extension().create_image_api() };
        let inner = unsafe { (api.read.unwrap())(path.as_ptr(), format) };
        Self {
            _api: api,
            _inner: inner,
        }
    }
    pub fn write(&self, path: impl AsRef<str>) -> bool {
        let path = std::ffi::CString::new(path.as_ref()).unwrap();
        unsafe { (self._api.write.unwrap())(path.as_ptr(), &self._inner) }
    }
}
impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            (self._api.destroy_image.unwrap())(&self._inner);
        }
    }
}
