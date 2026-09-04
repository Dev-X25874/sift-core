//! In-memory sorted buffer that absorbs writes before they're flushed to an
//! sstable. Backed by a `BTreeMap<InternalKey, Value>` — a skiplist is the
//! textbook choice (lock-free concurrent inserts), but a BTreeMap gets the
//! same O(log n) sorted-insert behavior with less unsafe code, which is the
//! right tradeoff for a single-writer engine like this one.

use crate::{InternalKey, Seq, Value};
use std::collections::BTreeMap;

pub struct MemTable {
    entries: BTreeMap<InternalKey, Value>,
    approx_bytes: usize,
}

impl MemTable {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            approx_bytes: 0,
        }
    }

    pub fn put(&mut self, key: &[u8], val: &[u8], seq: Seq) {
        self.approx_bytes += key.len() + val.len() + 24; // rough per-entry overhead
        self.entries
            .insert(InternalKey::new(key, seq), Value::Put(val.to_vec()));
    }

    pub fn delete(&mut self, key: &[u8], seq: Seq) {
        self.approx_bytes += key.len() + 24;
        self.entries
            .insert(InternalKey::new(key, seq), Value::Delete);
    }

    /// Snapshot read: newest version of `key` with seq <= `as_of`.
    /// Because InternalKey sorts (user_key ASC, seq DESC), the first match
    /// in a forward range starting at (key, u64::MAX) is exactly that.
    pub fn get(&self, key: &[u8], as_of: Seq) -> Option<&Value> {
        let start = InternalKey::new(key, Seq::MAX);
        self.entries
            .range(start..)
            .take_while(|(ik, _)| ik.user_key == key)
            .find(|(ik, _)| ik.seq <= as_of)
            .map(|(_, v)| v)
    }

    pub fn approx_bytes(&self) -> usize {
        self.approx_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates all entries in on-disk sort order — exactly what an
    /// sstable writer needs to consume when flushing this memtable.
    pub fn iter(&self) -> impl Iterator<Item = (&InternalKey, &Value)> {
        self.entries.iter()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.approx_bytes = 0;
    }
}

impl Default for MemTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvcc_snapshot_read_sees_correct_version() {
        let mut mt = MemTable::new();
        mt.put(b"k", b"v1", 1);
        mt.put(b"k", b"v2", 2);
        mt.delete(b"k", 3);
        mt.put(b"k", b"v4", 4);

        assert_eq!(mt.get(b"k", 1), Some(&Value::Put(b"v1".to_vec())));
        assert_eq!(mt.get(b"k", 2), Some(&Value::Put(b"v2".to_vec())));
        assert_eq!(mt.get(b"k", 3), Some(&Value::Delete));
        assert_eq!(mt.get(b"k", 4), Some(&Value::Put(b"v4".to_vec())));
        assert_eq!(mt.get(b"missing", 4), None);
    }

    #[test]
    fn iteration_is_key_ascending_seq_descending() {
        let mut mt = MemTable::new();
        mt.put(b"b", b"1", 1);
        mt.put(b"a", b"1", 1);
        mt.put(b"a", b"2", 2);
        let order: Vec<(Vec<u8>, Seq)> = mt
            .iter()
            .map(|(ik, _)| (ik.user_key.clone(), ik.seq))
            .collect();
        assert_eq!(
            order,
            vec![(b"a".to_vec(), 2), (b"a".to_vec(), 1), (b"b".to_vec(), 1)]
        );
    }
}
