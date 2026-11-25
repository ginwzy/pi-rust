//! Transpile results cached by content hash, in memory and optionally on disk.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::{TranspileOptions, Transpiled, transpile};
use crate::error::JsError;

/// Bumped when the cached format or the transform changes.
const CACHE_VERSION: &str = concat!("pi-js-transpile-2-", env!("CARGO_PKG_VERSION"), "-oxc-0.153");
const ENTRIES: usize = 1024;
const BYTES: usize = 8 * 1024 * 1024;
const LANES: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key([u8; 32], Option<Arc<Path>>);

#[derive(Default)]
struct Memory {
    entries: HashMap<Key, (Arc<Transpiled>, usize)>,
    order: VecDeque<Key>,
    bytes: usize,
}

impl Memory {
    fn get(&self, key: &Key) -> Option<Arc<Transpiled>> {
        self.entries.get(key).map(|(value, _)| value.clone())
    }

    fn insert(&mut self, key: Key, value: Arc<Transpiled>) {
        let bytes = value.code.capacity() + value.map.as_ref().map_or(0, super::PositionMap::retained_bytes);
        if bytes > BYTES || self.entries.contains_key(&key) {
            return;
        }
        while self.entries.len() >= ENTRIES || self.bytes + bytes > BYTES {
            let old = self.order.pop_front().expect("nonempty cache");
            self.bytes -= self.entries.remove(&old).expect("queued cache entry").1;
        }
        self.bytes += bytes;
        self.order.push_back(key.clone());
        self.entries.insert(key, (value, bytes));
    }
}

static MEMORY: std::sync::LazyLock<Mutex<Memory>> = std::sync::LazyLock::new(Mutex::default);
static MISS: [Mutex<()>; LANES] = [const { Mutex::new(()) }; LANES];

/// Thread-safe transpile cache. Entries are keyed by the SHA-256 of the source text, the file
/// extension, and the options, so a renamed file with the same content reuses its entry.
#[derive(Debug, Default)]
pub struct TranspileCache {
    options: TranspileOptions,
    dir: Option<Arc<Path>>,
}

impl TranspileCache {
    /// A cache kept in memory, and also in `dir` when one is given.
    pub fn new(options: TranspileOptions, dir: Option<PathBuf>) -> Self {
        Self { options, dir: dir.map(Into::into) }
    }

    pub fn options(&self) -> &TranspileOptions {
        &self.options
    }

    /// The cached result for this source, transpiling it on a miss.
    pub fn get_or_transpile(&self, path: &Path, source: &str) -> Result<Arc<Transpiled>, JsError> {
        let key = self.key(path, source);
        let disk = Key(key.0, self.dir.clone());
        let lookup = || {
            let memory = MEMORY.lock().unwrap_or_else(|poison| poison.into_inner());
            memory.get(&key).or_else(|| memory.get(&disk))
        };
        if let Some(hit) = lookup() {
            return Ok(hit);
        }
        let _miss = MISS[usize::from(key.0[0]) % LANES].lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(hit) = lookup() {
            return Ok(hit);
        }
        let hex = self.dir.as_ref().map(|_| key.0.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
        let (key, result) = match hex.as_deref().and_then(|hex| self.read_disk(hex)) {
            // Disk artifacts are scoped to their cache directory; only in-process output is global.
            Some(cached) => (disk, cached),
            None => {
                let fresh = transpile(path, source, &self.options)?;
                if let Some(hex) = &hex {
                    self.write_disk(hex, &fresh);
                }
                (key, fresh)
            }
        };
        let result = Arc::new(result);
        MEMORY.lock().unwrap_or_else(|poison| poison.into_inner()).insert(key, result.clone());
        Ok(result)
    }

    /// Number of entries held by the shared in-process cache.
    pub fn len(&self) -> usize {
        MEMORY.lock().unwrap_or_else(|poison| poison.into_inner()).entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn key(&self, path: &Path, source: &str) -> Key {
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or_default();
        let mut context = ring::digest::Context::new(&ring::digest::SHA256);
        for part in [CACHE_VERSION, &self.options.cache_key(), extension] {
            context.update(part.as_bytes());
            context.update(&[0]);
        }
        if path.to_string_lossy().ends_with(".d.ts") {
            context.update(b"declaration\0");
        }
        context.update(source.as_bytes());
        Key(context.finish().as_ref().try_into().expect("SHA-256 digest"), None)
    }

    fn read_disk(&self, key: &str) -> Option<Transpiled> {
        let text = fs::read_to_string(self.dir.as_ref()?.join(format!("{key}.json"))).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn write_disk(&self, key: &str, value: &Transpiled) {
        let Some(dir) = &self.dir else { return };
        let Ok(text) = serde_json::to_string(value) else { return };
        if fs::create_dir_all(dir).is_err() {
            return;
        }
        // Write then rename so a concurrent reader never sees a partial entry.
        let temp = dir.join(format!("{key}.{}.tmp", std::process::id()));
        if fs::write(&temp, text).is_ok() && fs::rename(&temp, dir.join(format!("{key}.json"))).is_err() {
            let _ = fs::remove_file(&temp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_bounds_count_and_payload_bytes() {
        let mut cache = Memory::default();
        let value = Arc::new(Transpiled {
            code: "0".into(),
            format: crate::transpile::ModuleFormat::Esm,
            free_globals: Default::default(),
            map: None,
        });
        let key = |index: usize| {
            let mut bytes = [0; 32];
            bytes[..size_of::<usize>()].copy_from_slice(&index.to_ne_bytes());
            Key(bytes, None)
        };
        for index in 0..=ENTRIES {
            cache.insert(key(index), value.clone());
        }
        assert_eq!(cache.entries.len(), ENTRIES);
        assert!(cache.get(&key(0)).is_none());

        let large = Arc::new(Transpiled { code: "z".repeat(BYTES), ..(*value).clone() });
        cache.insert(key(ENTRIES + 1), large);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.bytes, BYTES);
        let oversized = Arc::new(Transpiled { code: "z".repeat(BYTES + 1), ..(*value).clone() });
        cache.insert(key(ENTRIES + 2), oversized);
        assert!(cache.get(&key(ENTRIES + 2)).is_none());
    }
}
