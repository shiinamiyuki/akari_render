#[allow(dead_code)]
use std::{
    any::Any,
    cell::UnsafeCell,
    io::{Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    sync::{atomic::AtomicBool, Arc},
};
use std::{cell::Cell, sync::atomic::AtomicUsize};

use parking_lot::{Mutex, RwLock};
use tempfile::tempfile;

pub struct VArray<T: Copy + Sync + Send> {
    base: usize,
    len: usize,
    page_size: usize,
    vmem: Arc<VArrayMem>,
    phandom: PhantomData<T>,
}
pub struct VArrayIterator<'a, T: Copy + Sync + Send + 'static> {
    array: &'a VArray<T>,
    index: usize,
}
impl<'a, T: Copy + Sync + Send + 'static> Iterator for VArrayIterator<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.array.len {
            let ret = Some(self.array.read(self.index));
            self.index += 1;
            ret
        } else {
            None
        }
    }
}
impl<T: Copy + Sync + Send + 'static> VArray<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn read(&self, index: usize) -> T {
        assert!(self.vmem.comitted);
        let page = index / self.page_size;
        let offset = index % self.page_size;
        self.vmem.read(self.base + page, offset)
    }
    pub fn iter<'a>(&'a self) -> VArrayIterator<'a, T> {
        VArrayIterator {
            array: self,
            index: 0,
        }
    }
}

struct PageData<T: Copy + Sync + Send>(Vec<T>);
trait TPageData: Send + Sync {
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
    data: Option<Box<dyn TPageData>>,
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
pub struct VArrayMem {
    pagetable: Vec<PageTableEntry>,
    pages: Vec<RwLock<Page>>,
    mutex: RwLock<MutexData>,
    page_size: usize,
    comitted: bool,
    npages: usize,
    max_mem: usize,
}
pub struct VArrayMemBuilder {
    vmem: Arc<VArrayMem>,
}
impl VArrayMemBuilder {
    pub fn new(max_mem: usize, page_size: usize) -> Self {
        let npages = (max_mem + page_size - 1) / page_size;
        let vmem = Arc::new(VArrayMem {
            pages: vec![],
            npages,
            pagetable: vec![],
            mutex: RwLock::new(MutexData { clock_hand: 0 }),
            page_size,
            comitted: false,
            max_mem,
        });
        Self { vmem }
    }
    pub fn build(self) -> Arc<VArrayMem> {
        unsafe {
            let vmem = &mut *(Arc::as_ptr(&self.vmem) as *mut VArrayMem);
            vmem.comitted = true;
        }
        log::info!(
            "VArrayMemBuilder: max_mem:{}MB, active pages:{}, max active pages:{}, total pages:{}",
            self.vmem.max_mem / (1024 * 1024),
            self.vmem.pages.len(),
            self.vmem.npages,
            self.vmem.pagetable.len()
        );
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
            if self.pages.len() < self.npages {
                self.pages.push(RwLock::new(Page {
                    data: None,
                    pte_idx: None,
                    referenced: AtomicBool::new(false),
                }));
            }
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
        // println!(
        //     "finding page, page count:{}, max page:{}",
        //     self.pages.len(),
        //     self.npages
        // );
        let hand = &mut data.clock_hand;
        loop {
            if *hand >= self.pages.len() {
                *hand = 0;
            }
            let page = self.pages[*hand].read();
            if page.referenced.load(std::sync::atomic::Ordering::Acquire) && page.pte_idx.is_some()
            {
                page.referenced
                    .store(false, std::sync::atomic::Ordering::Release);
            } else {
                // println!("found page");
                // dbg!(*hand);
                let idx = *hand;
                *hand += 1;
                // evict page
                if let Some(pte_idx) = page.pte_idx {
                    // println!("evict");
                    std::mem::drop(page);
                    self.unload_page(pte_idx);
                }
                // println!("done");
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
                page.data = Some(Box::new(PageData(data)));
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
        // println!("a");
        match &mut *status {
            PageTableEntryStatus::Mapped(physical_page_idx) => {
                // dbg!(*physical_page_idx);
                let mut page = self.pages[*physical_page_idx].write();
                // println!("b");
                let data = page.data.as_mut().unwrap().as_mut().as_u8();
                let mut swap_file = tempfile().unwrap();
                swap_file.write_all(data).unwrap();
                page.pte_idx = None;
                page.data = None;
                *status = PageTableEntryStatus::Unloaded(swap_file);
            }
            _ => unreachable!(),
        }
    }
    fn write_page<T: Copy + Sync + Send + 'static>(&mut self, pte_idx: usize, data: &[T]) {
        let pte = &self.pagetable[pte_idx];
        assert!(pte.size == data.len());

        let mut status = pte.status.write();
        match &mut *status {
            PageTableEntryStatus::Mapped(physical_page_idx) => {
                let mut page = self.pages[*physical_page_idx].write();
                // let pagedata = page
                //     .data
                //     .as_mut()
                //     .unwrap()
                //     .as_mut()
                //     .as_any_mut()
                //     .downcast_mut::<PageData<T>>()
                //     .unwrap();
                // pagedata.0 = data.to_vec();
                page.data = Some(Box::new(PageData(data.to_vec())));
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
    pub fn read<T: Copy + Sync + Send + 'static>(&self, pte_idx: usize, offset: usize) -> T {
        for _ in 0..2 {
            let mt_data = self.mutex.read();
            if let Some(page_idx) = self.pagetable[pte_idx].loaded() {
                let page = self.pages[page_idx].read();
                page.referenced
                    .store(true, std::sync::atomic::Ordering::Release);
                return page
                    .data
                    .as_ref()
                    .unwrap()
                    .as_any()
                    .downcast_ref::<PageData<T>>()
                    .unwrap()
                    .0[offset];
            } else {
                std::mem::drop(mt_data);
                self.load_page::<T>(pte_idx);
            }
        }
        unreachable!()
    }
}
