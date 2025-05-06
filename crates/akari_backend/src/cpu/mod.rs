use std::cell::RefCell;

use crate::*;
struct CpuSubmission {
    funcs: RefCell<Vec<Box<dyn FnMut()>>>,
}
impl AsAny for CpuSubmission {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
struct CpuSubmissionToken<'a> {
    
}
impl RawSubmission for CpuSubmission {
    fn submit<'a>(&self) -> Box<dyn SubmissionToken<'a>> {
        Box::new(CpuSubmissionToken {
            funcs: self.funcs.borrow().clone(),
        })
    }
}
impl CpuSubmission {
    fn push<F: FnMut() + 'static>(&self, f: F) {
        self.funcs.borrow_mut().push(Box::new(f));
    }
}
struct CpuRawBuffer {
    size: usize,
    align: usize,
    data: *mut u8,
}
impl RawBuffer for CpuRawBuffer {
    fn native_handle(&self) -> Option<u64> {
        Some(self.data as u64)
    }

    fn upload<'a>(&self, submission: &Submission<'a>, range: Range<usize>, data: &'a [u8]) {
        let sub = submission.get_inner::<CpuSubmission>();
        let ptr = data.as_ptr();
        let self_data_ptr = self.data;
        sub.push(move || unsafe {
            std::ptr::copy_nonoverlapping(
                ptr,
                self_data_ptr.add(range.start),
                range.end - range.start,
            );
        });
    }
    fn download<'a>(&self, submission: &Submission<'a>, range: Range<usize>, data: &'a mut [u8]) {
        let sub = submission.get_inner::<CpuSubmission>();
        let ptr = data.as_mut_ptr();
        let self_data_ptr = self.data;
        sub.push(move || unsafe {
            std::ptr::copy_nonoverlapping(
                self_data_ptr.add(range.start),
                ptr,
                range.end - range.start,
            );
        });
    }
}
impl Drop for CpuRawBuffer {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.data,
                std::alloc::Layout::from_size_align(self.size, self.align).unwrap(),
            );
        }
    }
}
pub struct CpuBackend {}
impl Backend for CpuBackend {
    fn create_raw_buffer(&self, size: usize, align: usize) -> Arc<dyn RawBuffer> {
        let data =
            unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(size, align).unwrap()) };
        assert!(
            !data.is_null(),
            "failed to allocate memory of {:?} bytes",
            size
        );
        Arc::new(CpuRawBuffer { size, align, data })
    }

    fn create_stream(&self) -> Arc<dyn Stream> {
        todo!()
    }

    fn create_raw_kernel(&self, src: &str) -> Arc<dyn RawKernel> {
        todo!()
    }
}
