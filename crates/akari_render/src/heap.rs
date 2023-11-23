use crate::*;
use luisa::resource::IoTexel;
use parking_lot::Mutex;
use std::{
    ops::Deref,
    sync::atomic::{AtomicBool, AtomicUsize},
};

/// *One heap to rule them all.*
pub struct MegaHeap {
    #[allow(dead_code)]
    device: Device,
    bindless: BindlessArray,
    buffer_cnt: AtomicUsize,
    tex2d_cnt: AtomicUsize,
    tex3d_cnt: AtomicUsize,
    dirty: AtomicBool,
    mutex: Mutex<()>,
}

impl MegaHeap {
    pub fn new(device: Device, count: usize) -> Self {
        assert!(count <= 50 * 10000, "Too many resources");
        let bindless = device.create_bindless_array(count);
        Self {
            device,
            bindless,
            buffer_cnt: AtomicUsize::new(0),
            tex2d_cnt: AtomicUsize::new(0),
            tex3d_cnt: AtomicUsize::new(0),
            dirty: AtomicBool::new(false),
            mutex: Mutex::new(()),
        }
    }
    pub fn alloc_buffer<T: Value>(&self, len: usize) -> (Buffer<T>, u32) {
        let _lk = self.mutex.lock();
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let buffer = self.device.create_buffer::<T>(len);
        let index = self
            .buffer_cnt
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bindless.emplace_buffer_async(index, &buffer);

        (buffer, index as u32)
    }
    pub fn bind_buffer<T: Value>(&self, buffer: &Buffer<T>) -> u32 {
        let _lk = self.mutex.lock();
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let index = self
            .buffer_cnt
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bindless.emplace_buffer_async(index, &buffer);
        index as u32
    }
    pub fn alloc_tex2d<T: IoTexel>(
        &self,
        storage: PixelStorage,
        width: u32,
        height: u32,
        mips: u32,
        sampler: TextureSampler,
    ) -> (Tex2d<T>, u32) {
        let _lk = self.mutex.lock();
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let tex = self.device.create_tex2d::<T>(storage, width, height, mips);
        let index = self
            .tex2d_cnt
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bindless.emplace_tex2d_async(index, &tex, sampler);
        (tex, index as u32)
    }
    pub fn bind_tex2d<T: IoTexel>(&self, tex: &Tex2d<T>, sampler: TextureSampler) -> u32 {
        let _lk = self.mutex.lock();
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let index = self
            .tex2d_cnt
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bindless.emplace_tex2d_async(index, &tex, sampler);
        index as u32
    }
    pub fn alloc_tex3d<T: IoTexel>(
        &self,
        storage: PixelStorage,
        width: u32,
        height: u32,
        depth: u32,
        mips: u32,
        sampler: TextureSampler,
    ) -> (Tex3d<T>, u32) {
        let _lk = self.mutex.lock();
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let tex = self
            .device
            .create_tex3d::<T>(storage, width, height, depth, mips);
        let index = self
            .tex3d_cnt
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bindless.emplace_tex3d_async(index, &tex, sampler);
        (tex, index as u32)
    }
    pub fn bind_tex3d<T: IoTexel>(&self, tex: &Tex3d<T>, sampler: TextureSampler) -> u32 {
        let _lk = self.mutex.lock();
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        let index = self
            .tex3d_cnt
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bindless.emplace_tex3d_async(index, &tex, sampler);
        index as u32
    }
    pub fn reset(&self) {
        let _lk = self.mutex.lock();
        for i in 0..self.buffer_cnt.load(std::sync::atomic::Ordering::Relaxed) {
            self.bindless.remove_buffer_async(i);
        }
        for i in 0..self.tex2d_cnt.load(std::sync::atomic::Ordering::Relaxed) {
            self.bindless.remove_tex2d_async(i);
        }
        for i in 0..self.tex3d_cnt.load(std::sync::atomic::Ordering::Relaxed) {
            self.bindless.remove_tex3d_async(i);
        }
        self.buffer_cnt
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.tex2d_cnt
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.tex3d_cnt
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn commit(&self) {
        let _lk = self.mutex.lock();
        if self.dirty.load(std::sync::atomic::Ordering::Relaxed) {
            self.bindless.update();
            self.dirty
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }
    pub fn var(&self) -> BindlessArrayVar {
        assert!(
            !self.dirty.load(std::sync::atomic::Ordering::Relaxed),
            "MegaHeap is dirty"
        );
        self.bindless.var()
    }
}
impl Deref for MegaHeap {
    type Target = BindlessArrayVar;
    fn deref(&self) -> &Self::Target {
        self.bindless.deref()
    }
}
