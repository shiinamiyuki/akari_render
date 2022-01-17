use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{parallel_for, UnsafePointer};

struct Sort<'a, T, F> {
    slice: &'a mut [T],
    tmp: Vec<T>,
    key: F,
    buckets: usize,
    bits: u64,
}
impl<'a, T: 'static + Send + Sync + Clone + Copy, F: Fn(&T) -> u64 + Send + Sync> Sort<'a, T, F> {
    fn round(&mut self, r: u64) {
        self.tmp.clear();
        self.tmp.extend_from_slice(self.slice);
        let parts = rayon::current_num_threads();
        let counts: Vec<RwLock<Vec<usize>>> = (0..parts)
            .map(|_| RwLock::new(vec![0; self.buckets]))
            .collect();
        let p_slice = UnsafePointer::new(self.slice.as_mut_ptr());
        let chunk = (self.slice.len() + parts - 1) / parts;
        let len = self.slice.len();
        let tmp = &self.tmp;
        let compute_bucket = |item| {
            let k = (self.key)(item);
            let mask = (1 << self.bits) - 1u64;
            let k = k >> r;
            let k = k & mask;
            k
        };
        parallel_for(parts, 1, |p| {
            let slice = unsafe { std::slice::from_raw_parts(p_slice.p, tmp.len()) };
            let mut counts = counts[p].write();
            for i in p * chunk..((p + 1) * chunk).min(len) {
                let item = &slice[i];
                let b = compute_bucket(item);
                counts[b as usize] += 1;
            }
        });
        let offsets: Vec<RwLock<Vec<usize>>> = (0..parts)
            .map(|_| RwLock::new(vec![0; self.buckets]))
            .collect();
        let mut base = 0;
        for b in 0..self.buckets {
            for k in 0..parts {
                let mut offsets = offsets[k].write();
                let cnt = counts[k].read();
                let c = cnt[b];
                offsets[b] += base;
                base += c;
            }
        }

        assert_eq!(base, self.slice.len());
        parallel_for(parts, 1, |p| {
            let mut offsets = offsets[p].write();
            let slice = unsafe { std::slice::from_raw_parts_mut(p_slice.p, self.tmp.len()) };
            for i in p * chunk..((p + 1) * chunk).min(len) {
                let item = &self.tmp[i];
                let b = compute_bucket(item) as usize;
                slice[offsets[b]] = *item;
                offsets[b] += 1;
            }
        });
    }
    pub fn run(&mut self) {
        assert_eq!(64 % self.bits, 0);
        let mut r = 0;
        while r < 64 {
            self.round(r);
            r += self.bits;
        }
    }
}
pub fn par_radix_sort_by<
    T: 'static + Send + Sync + Clone + Copy,
    F: Fn(&T) -> u64 + Send + Sync,
>(
    slice: &mut [T],
    key: F,
) {
    let bits = 4;
    let buckets = 1u64 << bits;
    let len = slice.len();
    let mut sort = Sort {
        slice,
        key,
        bits,
        buckets: buckets as usize,
        tmp: Vec::with_capacity(len),
    };
    sort.run();
}

mod test {

    #[test]
    fn test_simple() {
        let mut v = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];
        super::par_radix_sort_by(v.as_mut_slice(), |x| *x as u64);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
    #[test]
    fn test_large() {
        use super::*;
        use rand::thread_rng;
        use rand::Rng;
        use rayon::slice::ParallelSliceMut;
        let mut rng = thread_rng();
        let mut v1: Vec<u64> = (0..10000000).map(|_| rng.gen::<u64>()).collect();
        let mut v2 = v1.clone();
        par_radix_sort_by(v1.as_mut_slice(), |x| *x as u64);
        v2.par_sort();
        assert_eq!(v1, v2);
    }
}
