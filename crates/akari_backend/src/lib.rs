use std::{any::Any, ops::Range, sync::Arc};
pub mod cpu;
pub mod vk;
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
}
pub trait RawBuffer {
    fn native_handle(&self) -> Option<u64>;
    fn upload<'a>(&self, submission: &Submission<'a>, range: Range<usize>, data: &'a [u8]);
    fn download<'a>(&self, submission: &Submission<'a>, range: Range<usize>, data: &'a mut [u8]);
}

pub struct RawBufferView {
    pub raw_buffer: Arc<dyn RawBuffer>,
    pub offset: usize,
    pub size: usize,
}
pub struct Buffer<T: Copy> {
    pub raw_buffer: Arc<dyn RawBuffer>,
    pub size: usize,
    pub align: usize,
    marker: std::marker::PhantomData<T>,
}
pub trait SubmissionToken<'a> {
    fn wait(&self);
    fn completed(&self) -> bool;
}
pub trait RawSubmission: AsAny {
    fn submit<'a>(&self) -> Box<dyn SubmissionToken<'a>>;
}
pub struct Submission<'a> {
    pub(crate) raw_submission: Box<dyn RawSubmission>,
    marker: std::marker::PhantomData<&'a ()>,
}
impl<'a> Submission<'a> {
    pub fn submit(&self) -> Box<dyn SubmissionToken<'a>> {
        self.raw_submission.submit()
    }
    pub(crate) fn get_inner<T: Any>(&self) -> &T {
        self.raw_submission.as_any().downcast_ref::<T>().unwrap()
    }
}
pub trait Stream {
    fn new_submission<'a>(&self) -> Submission<'a>;
    fn native_handle(&self) -> Option<u64>;
    fn sync(&self);
}

pub trait RawKernel {}
pub struct Kernel<F> {
    pub raw_kernel: Box<dyn RawKernel>,
    marker: std::marker::PhantomData<F>,
}
pub struct Accel {}
pub trait Backend {
    fn create_raw_buffer(&self, size: usize, align: usize) -> Arc<dyn RawBuffer>;
    fn create_stream(&self) -> Arc<dyn Stream>;
    /// create a kernel from luisa-python source code
    fn create_raw_kernel(&self, src: &str) -> Arc<dyn RawKernel>;
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
