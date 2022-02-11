use crate::*;
use std::sync::atomic::{AtomicBool, AtomicPtr};
#[repr(align(64))]
struct AtomicFlags(AtomicBool);

pub struct Rcu<T> {
    flags: Vec<AtomicFlags>,
    value: AtomicPtr<T>,
}
pub struct Critical<'a, T> {
    rcu: &'a Rcu<T>,
}
impl<'a, T> Drop for Critical<'a, T> {
    fn drop(&mut self) {
        self.rcu.flags[rayon::current_thread_index().unwrap()]
            .0
            .store(false, Ordering::Release);
    }
}
impl<T> Rcu<T>
where
    T: Sync + Send,
{
    pub fn new(value: T) -> Self {
        Self {
            flags: (0..rayon::current_num_threads())
                .map(|_| AtomicFlags(AtomicBool::new(false)))
                .collect(),
            value: AtomicPtr::new(Box::into_raw(Box::new(value))),
        }
    }
    pub fn lock<'a>(&'a self) -> Critical<'a, T> {
        self.flags[rayon::current_thread_index().unwrap()]
            .0
            .store(true, Ordering::Release);
        Critical { rcu: self }
    }
    pub fn synchronize(&self) {
        for i in 0..rayon::current_num_threads() {
            let flag = &self.flags[i];
            while flag.0.load(Ordering::Acquire) {}
        }
    }
    pub fn replace(&self, value: T) {
        let ptr = Box::into_raw(Box::new(value));
        let old = self.value.swap(ptr, Ordering::AcqRel);
        self.synchronize();
        unsafe {
            Box::from_raw(old);
        }
    }
}
