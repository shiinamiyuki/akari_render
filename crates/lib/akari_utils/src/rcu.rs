// use crate::*;
// use std::sync::atomic::{AtomicBool, AtomicPtr};
// #[repr(align(64))]
// struct AtomicFlags(AtomicBool);

// pub struct Rcu<T> {
//     flags: Vec<AtomicFlags>,
//     value: AtomicPtr<T>,
// }
// impl<T> Rcu<T>
// where
//     T: Sync + Send,
// {
//     pub fn new(value: T) -> Self {
//         Self {
//             flags: vec![AtomicFlags(AtomicBool::new(false)); rayon::current_num_threads()],
//             value: AtomicPtr::new(Box::into_raw(Box::new(value))),
//         }
//     }
//     pub fn synchronize(&self){
//         for i in 0..rayon::current_num_threads() {
            
//         }
//     }
// }
