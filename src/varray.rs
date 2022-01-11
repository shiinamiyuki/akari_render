#[allow(dead_code)]
use std::{
    any::Any,
    cell::UnsafeCell,
    io::{Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    sync::{atomic::AtomicBool, Arc},
};

use parking_lot::{Mutex, RwLock};
use tempfile::tempfile;

pub struct VArray<T: Copy + Sync + Send> {
    base: usize,
    len: usize,
    page_size: usize,
    vmem: Arc<VArrayMem>,
    phandom: PhantomData<T>,
}
impl<T: Copy + Sync + Send + 'static> VArray<T> {
    pub fn read(&self, index: usize) -> T {
        assert!(self.vmem.comitted);
        let page = index / self.page_size;
        let offset = index % self.page_size;
        self.vmem.read(self.base + page, offset)
    }
}

struct PageData<T: Copy + Sync + Send>(Vec<T>);
trait TPageData {
    fn as_u8(&self) -> &[u8];
    fn as_any<'a>(&'a self) -> &'a dyn Any;
    fn as_any_mut<'a>(&'a mut self) -> &'a mut dyn Any;
}
impl<T: Copy + Sync + Send + 'static> TPageData for PageData<T> {
    fn as_u8(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const u8,
                self.0.len() * std::mem::size_of::<T>(),
            )
        }
    }
    fn as_any<'a>(&'a self) -> &'a dyn Any {
        self as &dyn Any
    }
    fn as_any_mut<'a>(&'a mut self) -> &'a mut dyn Any {
        self as &mut dyn Any
    }
}
struct Page {
    data: Box<dyn TPageData>,
    pte_idx: Option<usize>,
    referenced: AtomicBool,
}

enum PageTableEntryStatus {
    Mapped(usize),
    Unloaded(std::fs::File),
}
pub struct PageTableEntry {
    status: RwLock<PageTableEntryStatus>,
    size: usize,
}
impl PageTableEntry {
    pub fn loaded(&self) -> Option<usize> {
        let status = self.status.read();
        match &*status {
            PageTableEntryStatus::Mapped(i) => Some(*i),
            _ => None,
        }
    }
}
struct MutexData {
    clock_hand: usize,
}
struct VArrayMem {
    pagetable: Vec<PageTableEntry>,
    pages: Vec<RwLock<Page>>,
    mutex: RwLock<MutexData>,
    page_size: usize,
    comitted: bool,
}
struct VArrayMemBuilder {
    vmem: Arc<VArrayMem>,
}
impl VArrayMemBuilder {
    pub fn new(max_mem: usize, page_size: usize) -> Self {
        let vmem = Arc::new(VArrayMem {
            pages: vec![],
            pagetable: vec![],
            mutex: RwLock::new(MutexData { clock_hand: 0 }),
            page_size,
            comitted: false,
        });
        Self { vmem }
    }
    pub fn build(self) -> Arc<VArrayMem> {
        unsafe {
            let vmem = &mut *(Arc::as_ptr(&self.vmem) as *mut VArrayMem);
            vmem.comitted = true;
        }
        self.vmem
    }
    pub fn allocate<T: Copy + Sync + Send + 'static>(&mut self, data: &[T]) -> VArray<T> {
        unsafe {
            let vmem = &mut *(Arc::as_ptr(&self.vmem) as *mut VArrayMem);
            vmem.allocate(data, self.vmem.clone())
        }
    }
}
impl VArrayMem {
    fn allocate<T: Copy + Sync + Send + 'static>(
        &mut self,
        data: &[T],
        pself: Arc<VArrayMem>,
    ) -> VArray<T> {
        let page_size = (self.page_size + std::mem::size_of::<T>() - 1) / std::mem::size_of::<T>();
        let npages = (data.len() + page_size - 1) / page_size;
        let pte_base = self.pagetable.len();

        for i in 0..npages {
            let range = i * page_size..((i + 1) * page_size).min(data.len());
            let mut mt_data = self.mutex.write();
            let page_idx = self.locate_page(&mut mt_data);
            {
                let mut page = self.pages[page_idx].write();
                page.pte_idx = Some(pte_base + i);
                page.referenced
                    .store(true, std::sync::atomic::Ordering::Release);
            }

            let pte = PageTableEntry {
                status: RwLock::new(PageTableEntryStatus::Mapped(page_idx)),
                size: range.end - range.start,
            };
            self.pagetable.push(pte);
            std::mem::drop(mt_data);
            self.write_page(pte_base + i, &data[range]);
        }
        VArray {
            base: pte_base,
            len: data.len(),
            phandom: PhantomData {},
            page_size,
            vmem: pself,
        }
    }
    fn locate_page(&self, data: &mut MutexData) -> usize {
        let hand = &mut data.clock_hand;
        loop {
            if *hand >= self.pages.len() {
                *hand = 0;
            }
            let page = &self.pages[*hand].read();
            if page.referenced.load(std::sync::atomic::Ordering::Acquire) {
                page.referenced
                    .store(false, std::sync::atomic::Ordering::Release);
            } else {
                let idx = *hand;
                *hand += 1;
                // evict page
                if let Some(pte_idx) = page.pte_idx {
                    self.unload_page(pte_idx);
                }
                return idx;
            }
            *hand += 1;
        }
    }
    fn load_page<T: Copy + Sync + Send + 'static>(&self, pte_idx: usize) {
        let mut mt_data = self.mutex.write();
        let pte = &self.pagetable[pte_idx];
        let mut status = pte.status.write();
        let page_idx = match &mut *status {
            PageTableEntryStatus::Unloaded(swap_file) => {
                swap_file.seek(SeekFrom::Start(0)).unwrap();
                let mut buffer = Vec::<u8>::new();
                swap_file.read_to_end(&mut buffer).unwrap();
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        buffer.as_ptr() as *const T,
                        buffer.len() / std::mem::size_of::<T>(),
                    )
                };
                let data = slice.to_vec();
                let free_page = self.locate_page(&mut mt_data);
                let mut page = self.pages[free_page].write();
                page.referenced
                    .store(true, std::sync::atomic::Ordering::Release);
                page.pte_idx = Some(pte_idx);
                page.data = Box::new(PageData(data));
                free_page
            }
            _ => {
                return;
            }
        };
        *status = PageTableEntryStatus::Mapped(page_idx);
    }
    fn unload_page(&self, pte_idx: usize) {
        let pte = &self.pagetable[pte_idx];
        let mut status = pte.status.write();
        match &mut *status {
            PageTableEntryStatus::Mapped(physical_page_idx) => {
                let mut page = self.pages[*physical_page_idx].write();
                let data = page.data.as_mut().as_u8();
                let mut swap_file = tempfile().unwrap();
                swap_file.write_all(data).unwrap();
                *status = PageTableEntryStatus::Unloaded(swap_file);
            }
            _ => unreachable!(),
        }
    }
    fn write_page<T: Copy + Sync + Send + 'static>(&mut self, pte_idx: usize, data: &[T]) {
        let pte = &self.pagetable[pte_idx];
        assert!(pte.size == data.len());
        loop {
            let mut status = pte.status.write();
            match &mut *status {
                PageTableEntryStatus::Mapped(physical_page_idx) => {
                    let mut page = self.pages[*physical_page_idx].write();
                    let pagedata = page
                        .data
                        .as_mut()
                        .as_any_mut()
                        .downcast_mut::<PageData<T>>()
                        .unwrap();
                    pagedata.0 = data.to_vec();
                }
                PageTableEntryStatus::Unloaded(swap_file) => {
                    swap_file.seek(SeekFrom::Start(0)).unwrap();
                    swap_file
                        .write_all(unsafe {
                            std::slice::from_raw_parts(
                                data.as_ptr() as *const u8,
                                data.len() * std::mem::size_of::<T>(),
                            )
                        })
                        .unwrap();
                }
            }
        }
    }
    pub fn read<T: Copy + Sync + Send + 'static>(&self, pte_idx: usize, offset: usize) -> T {
        loop {
            let mt_data = self.mutex.read();
            if let Some(page_idx) = self.pagetable[pte_idx].loaded() {
                let page = self.pages[page_idx].read();
                page.referenced
                    .store(true, std::sync::atomic::Ordering::Release);
                return page.data.as_any().downcast_ref::<PageData<T>>().unwrap().0[offset];
            } else {
                std::mem::drop(mt_data);
                self.load_page::<T>(pte_idx);
            }
        }
    }
}
