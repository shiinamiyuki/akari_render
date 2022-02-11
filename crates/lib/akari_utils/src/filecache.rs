use std::collections::HashMap;
use std::fs;
use std::fs::Metadata;
use std::io::Result;
use std::path::{Path, PathBuf};
pub trait Resource: Sized {
    fn load(path: &Path) -> Result<Self>;
}
struct CacheEntry<R> {
    resource: R,
    metadata: Metadata,
}
pub struct FileCache<R> {
    cache: HashMap<PathBuf, CacheEntry<R>>,
}
impl<R: Resource + Clone> FileCache<R> {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
    fn check_cache_validity(&self, path: &PathBuf) -> Result<bool> {
        if let Some(entry) = self.cache.get(path) {
            let cached = &entry.metadata;
            let cur = fs::metadata(path)?;
            Ok(cur.modified()? == cached.modified()?)
        } else {
            Ok(false)
        }
    }
    pub fn load(&mut self, path: &Path) -> Result<R> {
        let path = PathBuf::from(path);
        if self.check_cache_validity(&path)? {
            Ok(self.cache.get(&path).unwrap().resource.clone())
        } else {
            // this is not safe
            let metadata = fs::metadata(path.clone())?;
            let resource = R::load(&path)?;
            self.cache.insert(
                path,
                CacheEntry {
                    resource: resource.clone(),
                    metadata,
                },
            );
            Ok(resource)
        }
    }
    pub fn invalidate(&mut self, path: &Path) {
        self.cache.remove(&PathBuf::from(path));
    }
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
