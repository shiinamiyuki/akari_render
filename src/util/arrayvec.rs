use std::{alloc::Layout, marker::PhantomData, ops::Range};
pub trait ArrayVecStorage<T> {
    fn ptr_mut(&self) -> *mut T;
    fn capacity(&self) -> usize;
}
impl<T> ArrayVecStorage<T> for Vec<T> {
    fn ptr_mut(&self) -> *mut T {
        self.as_ptr() as *mut T
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }
}
pub type DynStorage<T> = Box<dyn ArrayVecStorage<T>>;
impl<T> ArrayVecStorage<T> for DynStorage<T> {
    fn ptr_mut(&self) -> *mut T {
        self.as_ref().ptr_mut()
    }

    fn capacity(&self) -> usize {
        self.as_ref().capacity()
    }
}
pub struct VirtualStorage<T> {
    alloc: region::Allocation,
    capacity: usize,
    phantom: PhantomData<T>,
}
impl<T> VirtualStorage<T> {
    pub fn new(capacity: usize) -> Self {
        let alloc = region::alloc(
            std::mem::size_of::<T>() * capacity,
            region::Protection::READ_WRITE,
        )
        .unwrap();
        Self {
            alloc,
            capacity,
            phantom: PhantomData {},
        }
    }
}
impl<T> ArrayVecStorage<T> for VirtualStorage<T> {
    fn ptr_mut(&self) -> *mut T {
        self.alloc.as_ptr::<T>() as *mut T
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}
#[allow(dead_code)]
pub struct ArrayVec<T, S: ArrayVecStorage<T> = Vec<T>> {
    storage: S,
    ptr: *mut T,
    capacity: usize,
    len: usize,
}
unsafe impl<T, S: ArrayVecStorage<T>> Send for ArrayVec<T, S> {}
unsafe impl<T, S: ArrayVecStorage<T>> Sync for ArrayVec<T, S> {}
impl<T: Clone, S: ArrayVecStorage<T>> ArrayVec<T, S> {
    pub fn resize(&mut self, new_len: usize, val: T) {
        assert!(new_len <= self.capacity);
        if new_len < self.len {
            self.shrink(new_len);
        } else {
            for i in self.len..new_len {
                unsafe {
                    std::ptr::write(self.ptr.offset(i as isize), val.clone());
                }
            }
            self.len = new_len;
        }
    }
    pub fn extend_from_slice(&mut self, data: &[T]) {
        assert!(data.len() + self.len() <= self.capacity());
        for i in 0..data.len() {
            unsafe {
                std::ptr::write(self.ptr.offset((self.len + i) as isize), data[i].clone());
            }
        }
        self.len = data.len() + self.len;
    }
}
impl<T> ArrayVec<T, Vec<T>> {
    pub fn with_capacity(capacity: usize) -> Self {
        unsafe { Self::from_storage(Vec::with_capacity(capacity)) }
    }
}
impl<T, S: ArrayVecStorage<T>> ArrayVec<T, S> {
    pub unsafe fn from_storage(storage: S) -> Self {
        let ptr = storage.ptr_mut();
        let capacity = storage.capacity();
        Self {
            storage,
            ptr,
            len: 0,
            capacity,
        }
    }
    fn shrink(&mut self, new_len: usize) {
        assert!(new_len <= self.len);
        for i in new_len..self.len {
            unsafe {
                std::ptr::read(self.ptr.offset(i as isize) as *const T);
            }
        }
    }
    pub fn clear(&mut self){
        self.shrink(0);
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn push(&mut self, val: T) -> Result<(), ()> {
        if self.len + 1 > self.capacity {
            Err(())
        } else {
            unsafe {
                std::ptr::write(self.ptr.offset(self.len as isize), val);
                self.len += 1;
            }
            Ok(())
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(std::ptr::read(
                    self.ptr.offset(self.len as isize) as *const T
                ))
            }
        }
    }
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
    pub fn as_slice_mut(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl<T, S: ArrayVecStorage<T>> Drop for ArrayVec<T, S> {
    fn drop(&mut self) {
        self.shrink(0);
    }
}
impl<T, S: ArrayVecStorage<T>> std::ops::Deref for ArrayVec<T, S> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<T, S: ArrayVecStorage<T>> std::ops::DerefMut for ArrayVec<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}
