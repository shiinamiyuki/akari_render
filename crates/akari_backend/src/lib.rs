use std::{ops::Range, sync::Arc};

pub trait RawBuffer {
    fn native_handle(&self) -> Option<u64>;
    fn upload<'a>(&self, submission: &dyn Submission<'a>, range: Range<usize>, data: &'a [u8]);
    fn download<'a>(
        &self,
        submission: &dyn Submission<'a>,
        range: Range<usize>,
        data: &'a mut [u8],
    );
}
pub struct RawBufferView {
    pub raw_buffer: Box<dyn RawBuffer>,
    pub offset: usize,
    pub size: usize,
}
pub struct Buffer<T: Copy> {
    pub raw_buffer: Box<dyn RawBuffer>,
    pub size: usize,
    pub align: usize,
    marker: std::marker::PhantomData<T>,
}
pub trait Submission<'a> {
    fn submit(&self);
}
pub trait Stream {
    fn new_submission(&self) -> Box<dyn Submission>;
    fn native_handle(&self) -> Option<u64>;
}

pub trait RawKernel {}
pub struct Kernel<F> {
    pub raw_kernel: Box<dyn RawKernel>,
    marker: std::marker::PhantomData<F>,
}
struct Accel {

}
pub trait Backend {
    fn create_raw_buffer(&self, size: usize, align: usize) -> Box<dyn RawBuffer>;
    fn create_stream(&self) -> Box<dyn Stream>;
    /// create a kernel from luisa-python source code
    fn create_raw_kernel(&self, src: &str) -> Box<dyn RawKernel>;
}

pub struct Device {
    pub backend: Arc<dyn Backend>,
}
impl Device {
    pub fn create_buffer<T: Copy>(&self, size: usize) -> Buffer<T> {
        Buffer::<T> {
            raw_buffer: self
                .backend
                .create_raw_buffer(size, std::mem::align_of::<T>()),
            size,
            align: std::mem::align_of::<T>(),
            marker: std::marker::PhantomData,
        }
    }
}
