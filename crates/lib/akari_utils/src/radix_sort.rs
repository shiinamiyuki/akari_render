use crate::*;
use crate::{parallel_for, UnsafePointer};
use parking_lot::RwLock;

struct Sort<T, F> {
    src: UnsafePointer<T>,
    dst: UnsafePointer<T>,
    key: F,
    buckets: usize,
    len: usize,
    bits: u64,
    buffer: Vec<RwLock<Vec<T>>>,
    buffer_size: usize,
}
impl<T: 'static + Send + Sync + Clone + Copy + Default, F: Fn(&T) -> u64 + Send + Sync> Sort<T, F> {
    fn round(&mut self, r: u64, last: bool) {
        let parts = rayon::current_num_threads();
        let counts: Vec<RwLock<Vec<usize>>> = (0..parts)
            .map(|_| RwLock::new(vec![0; self.buckets]))
            .collect();
        let chunk = (self.len + parts - 1) / parts;
        let len = self.len;
        let mask = (1 << self.bits) - 1u64;
        let compute_bucket = |item| {
            let k = (self.key)(item);
            let k = k >> r;
            let k = k & mask;
            k
        };
        parallel_for(parts, 1, |p| {
            let src = unsafe { std::slice::from_raw_parts(self.src.as_ptr(), self.len) };
            // let mut dst = unsafe { std::slice::from_raw_parts_mut(self.dst, self.len) };
            let mut counts = counts[p].try_write().unwrap();
            for i in p * chunk..((p + 1) * chunk).min(len) {
                // let item = &slice[i];
                let item = unsafe { src.get_unchecked(i) };
                let b = compute_bucket(item);
                // counts[b as usize] += 1;
                unsafe {
                    *counts.get_unchecked_mut(b as usize) += 1;
                }
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

        assert_eq!(base, self.len);
        parallel_for(parts, 1, |p| {
            let src = unsafe { std::slice::from_raw_parts(self.src.as_ptr(), self.len) };
            let dst = unsafe { std::slice::from_raw_parts_mut(self.dst.as_ptr(), self.len) };
            let mut offsets = offsets[p].try_write().unwrap();
            let mut buf_cnt = vec![0usize; self.buckets];
            let mut buffer = self.buffer[p].try_write().unwrap();
            let hi = ((p + 1) * chunk).min(len);
            for i in p * chunk..hi {
                let item = unsafe { src.get_unchecked(i) }; //&self.tmp[i];
                let b = compute_bucket(item) as usize;
                // slice[offsets[b]] = *item;
                unsafe {
                    *buffer.get_unchecked_mut(b * self.buffer_size + *buf_cnt.get_unchecked(b)) =
                        *item;
                    *buf_cnt.get_unchecked_mut(b) += 1;
                    if *buf_cnt.get_unchecked(b) == self.buffer_size {
                        let cnt = *buf_cnt.get_unchecked(b);
                        let offset = *offsets.get_unchecked(b);
                        // dst[offset..offset + cnt].copy_from_slice(
                        // &buffer[b * self.buffer_size..b * self.buffer_size + cnt],
                        // );
                        std::ptr::copy_nonoverlapping(
                            buffer[b * self.buffer_size..b * self.buffer_size + cnt].as_ptr(),
                            dst[offset..offset + cnt].as_mut_ptr(),
                            cnt,
                        );
                        *offsets.get_unchecked_mut(b) += cnt;
                        *buf_cnt.get_unchecked_mut(b) = 0;
                    }
                    if i + 1 == hi {
                        for b in 0..self.buckets {
                            let cnt = *buf_cnt.get_unchecked(b);
                            let offset = *offsets.get_unchecked(b);
                            std::ptr::copy_nonoverlapping(
                                buffer[b * self.buffer_size..b * self.buffer_size + cnt].as_ptr(),
                                dst[offset..offset + cnt].as_mut_ptr(),
                                cnt,
                            );
                            *offsets.get_unchecked_mut(b) += cnt;
                        }
                        buf_cnt.fill(0);
                    }
                }

                // offsets[b] += 1;
            }
        });
        if !last {
            std::mem::swap(&mut self.src, &mut self.dst);
        }
    }
    pub fn run(&mut self) {
        // assert_eq!(64 % self.bits, 0);
        let mut r = 0;
        while r < 64 {
            self.round(r, r + self.bits >= 64);
            r += self.bits;
        }
    }
}
#[allow(dead_code)]
pub fn par_radix_sort_by<
    T: 'static + Send + Sync + Clone + Copy + Default,
    F: Fn(&T) -> u64 + Send + Sync,
>(
    slice: &mut [T],
    key: F,
) {
    let bits = 8;
    let buckets = 1u64 << bits;
    let mut tmp = slice.to_vec();
    let buffer_size = (128 * 1024 / buckets as usize / std::mem::size_of::<T>()).max(1);
    let mut sort = Sort {
        src: UnsafePointer::new(slice.as_mut_ptr()),
        dst: UnsafePointer::new(tmp.as_mut_ptr()),
        key,
        bits,
        len: slice.len(),
        buckets: buckets as usize,
        buffer_size,
        buffer: (0..rayon::current_num_threads())
            .map(|_| RwLock::new(vec![Default::default(); buffer_size * buckets as usize]))
            .collect(),
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
        use crate::profile_fn;
        use rand::thread_rng;
        use rand::Rng;
        use rayon::slice::ParallelSliceMut;
        let mut rng = thread_rng();
        let mut v1: Vec<u32> = (0..2000000).map(|_| rng.gen::<u32>()).collect();
        let mut v2 = v1.clone();

        let t1 = profile_fn(|| {
            par_radix_sort_by(v1.as_mut_slice(), |x| *x as u64);
        })
        .1;
        let t2 = profile_fn(|| {
            v2.par_sort_unstable();
        })
        .1;
        assert_eq!(v1, v2);
    }
}
