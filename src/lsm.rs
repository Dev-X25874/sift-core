//! Top-level LSM engine. Write path: WAL (durability) -> memtable (sorted
//! buffer) -> flush to sstable once the memtable crosses a size threshold ->
//! compact once too many sstables pile up. Read path: memtable -> tiers
//! newest-to-oldest, each guarded by its bloom filter.
//!
//! Single tier list rather than LevelDB-style leveling (L0..Ln with size
//! ratios) — leveling optimizes write amplification at gigabyte scale; at
//! this scale it'd add complexity without changing correctness reasoning.

use crate::compaction;
use crate::memtable::MemTable;
use crate::sstable::{SSTable, SSTableIter, SSTableMeta, SSTableWriter};
use crate::wal::{self, Wal};
use crate::{Seq, Value};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub struct LsmEngine {
    dir: PathBuf,
    wal: Wal,
    memtable: MemTable,
    tiers: Vec<SSTableMeta>, // index 0 = newest
    next_seq: Seq,
    next_sst_id: u64,
    flush_threshold_bytes: usize,
}

impl LsmEngine {
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let wal_path = dir.join("wal.log");

        // Crash recovery: replay whatever the WAL has (anything past the
        // last torn record is silently dropped by `wal::replay`).
        let mut memtable = MemTable::new();
        let mut max_seq: Seq = 0;
        for rec in wal::replay(&wal_path)? {
            match rec.op {
                wal::Op::Put => memtable.put(&rec.key, &rec.val, rec.seq),
                wal::Op::Delete => memtable.delete(&rec.key, rec.seq),
            }
            max_seq = max_seq.max(rec.seq);
        }
        let wal = Wal::open(&wal_path)?;

        // Discover already-flushed sstables: "L{id}.sst", newest (highest id) first.
        // Open each file to read the footer and get the real entry_count —
        // reconstructing with zeroed entry_count caused undersized bloom filters
        // after post-recovery compaction.
        //
        // Stale .sst.tmp files from an aborted flush or compaction are removed here.
        let mut tiers = Vec::new();
        let mut max_sst_id = 0u64;
        if let Ok(read_dir) = fs::read_dir(&dir) {
            let mut found: Vec<(u64, PathBuf)> = Vec::new();
            for entry in read_dir.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(".sst.tmp") {
                        fs::remove_file(&path).ok();
                        continue;
                    }
                    if let Some(id_str) =
                        name.strip_prefix('L').and_then(|s| s.strip_suffix(".sst"))
                    {
                        if let Ok(id) = id_str.parse::<u64>() {
                            max_sst_id = max_sst_id.max(id);
                            found.push((id, path));
                        }
                    }
                }
            }
            found.sort_by_key(|(id, _)| std::cmp::Reverse(*id));
            for (_, path) in found {
                let meta = SSTable::open(&path)?.into_meta();
                tiers.push(meta);
            }
        }

        Ok(Self {
            dir,
            wal,
            memtable,
            tiers,
            next_seq: max_seq + 1,
            next_sst_id: max_sst_id + 1,
            flush_threshold_bytes: 4 * 1024 * 1024,
        })
    }

    pub fn put(&mut self, key: &[u8], val: &[u8]) -> io::Result<Seq> {
        let seq = self.alloc_seq();
        self.wal.append(seq, wal::Op::Put, key, val)?;
        self.wal.sync()?;
        self.memtable.put(key, val, seq);
        self.maybe_flush()?;
        Ok(seq)
    }

    pub fn delete(&mut self, key: &[u8]) -> io::Result<Seq> {
        let seq = self.alloc_seq();
        self.wal.append(seq, wal::Op::Delete, key, &[])?;
        self.wal.sync()?;
        self.memtable.delete(key, seq);
        self.maybe_flush()?;
        Ok(seq)
    }

    pub fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        self.get_at(key, Seq::MAX)
    }

    /// MVCC snapshot read: the state of `key` as of write-sequence `as_of`.
    pub fn get_at(&self, key: &[u8], as_of: Seq) -> io::Result<Option<Vec<u8>>> {
        if let Some(v) = self.memtable.get(key, as_of) {
            return Ok(v.as_put().map(|b| b.to_vec()));
        }
        match compaction::lookup_in_tiers(&self.tiers, key, as_of)? {
            Some(Value::Put(v)) => Ok(Some(v)),
            Some(Value::Delete) | None => Ok(None),
        }
    }

    pub fn current_seq(&self) -> Seq {
        self.next_seq.saturating_sub(1)
    }

    fn alloc_seq(&mut self) -> Seq {
        let s = self.next_seq;
        self.next_seq += 1;
        s
    }

    fn maybe_flush(&mut self) -> io::Result<()> {
        if self.memtable.approx_bytes() >= self.flush_threshold_bytes {
            self.flush()?;
        }
        Ok(())
    }

    /// Flushes the memtable to a new sstable unconditionally (no-op if empty).
    /// Exposed so tests/CLI can control the durability/compaction boundary.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.memtable.is_empty() {
            return Ok(());
        }
        let id = self.next_sst_id;
        self.next_sst_id += 1;
        let final_path = self.dir.join(format!("L{}.sst", id));
        let tmp_path = self.dir.join(format!("L{}.sst.tmp", id));

        // Write to .tmp first, then rename atomically — same pattern as
        // compaction. A crash mid-write leaves a .sst.tmp that gets cleaned
        // up on the next open(), rather than a corrupt .sst that blocks
        // recovery (LsmEngine::open calls SSTable::open()?  on every .sst).
        let expected = (self.memtable.approx_bytes() / 48).max(1);
        let mut writer = SSTableWriter::create(&tmp_path, expected)?;
        for (ik, v) in self.memtable.iter() {
            writer.add(ik, v)?;
        }
        // finish() calls sync_data() before returning, so the SSTable is
        // durable on disk before we truncate the WAL below.
        let mut meta = writer.finish()?;
        fs::rename(&tmp_path, &final_path)?;
        meta.path = final_path;

        self.tiers.insert(0, meta);
        self.memtable.clear();
        self.wal = Wal::reset(self.dir.join("wal.log"))?;

        self.maybe_compact()
    }

    fn maybe_compact(&mut self) -> io::Result<()> {
        if !compaction::should_compact(self.tiers.len()) {
            return Ok(());
        }
        let inputs: Vec<SSTableMeta> = std::mem::take(&mut self.tiers);
        let id = self.next_sst_id;
        self.next_sst_id += 1;
        let path = self.dir.join(format!("L{}.sst", id));
        // Single-tier design: every compaction sees the whole dataset, so
        // tombstones can always be fully GC'd here.
        let merged = compaction::compact(&inputs, &path, true)?;
        for input in &inputs {
            fs::remove_file(&input.path).ok();
        }
        self.tiers.push(merged);
        Ok(())
    }

    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Full scan of the current snapshot's live (non-deleted) key/value pairs.
    /// Used by callers that keep derived in-memory state and need to rebuild
    /// it after a restart.
    ///
    /// "Newest version wins" via a hashmap keyed by user_key — simpler than
    /// a heap merge and fine at the scale this engine targets.
    pub fn iter_all(&self) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut latest: HashMap<Vec<u8>, (Seq, Value)> = HashMap::new();

        for meta in &self.tiers {
            for item in SSTableIter::open(&meta.path)? {
                let (ik, v) = item?;
                latest
                    .entry(ik.user_key)
                    .and_modify(|e| {
                        if ik.seq > e.0 {
                            *e = (ik.seq, v.clone());
                        }
                    })
                    .or_insert((ik.seq, v));
            }
        }
        for (ik, v) in self.memtable.iter() {
            latest
                .entry(ik.user_key.clone())
                .and_modify(|e| {
                    if ik.seq > e.0 {
                        *e = (ik.seq, v.clone());
                    }
                })
                .or_insert((ik.seq, v.clone()));
        }

        Ok(latest
            .into_iter()
            .filter_map(|(k, (_, v))| match v {
                Value::Put(bytes) => Some((k, bytes)),
                Value::Delete => None,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!(
            "sift_core_lsm_test_{}_{}",
            std::process::id(),
            name
        ));
        fs::remove_dir_all(&p).ok();
        p
    }

    #[test]
    fn put_get_delete_roundtrip() {
        let dir = tmp_dir("roundtrip");
        let mut db = LsmEngine::open(&dir).unwrap();
        db.put(b"a", b"1").unwrap();
        db.put(b"b", b"2").unwrap();
        assert_eq!(db.get(b"a").unwrap(), Some(b"1".to_vec()));
        db.delete(b"a").unwrap();
        assert_eq!(db.get(b"a").unwrap(), None);
        assert_eq!(db.get(b"b").unwrap(), Some(b"2".to_vec()));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flush_and_compaction_preserve_reads() {
        let dir = tmp_dir("flush_compact");
        let mut db = LsmEngine::open(&dir).unwrap();
        for i in 0..200u32 {
            db.put(
                format!("k{:05}", i).as_bytes(),
                format!("v{}", i).as_bytes(),
            )
            .unwrap();
            if i % 20 == 0 {
                db.flush().unwrap();
            }
        }
        for i in 0..200u32 {
            let got = db.get(format!("k{:05}", i).as_bytes()).unwrap();
            assert_eq!(got, Some(format!("v{}", i).into_bytes()));
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn crash_recovery_replays_wal() {
        let dir = tmp_dir("crash_recovery");
        {
            let mut db = LsmEngine::open(&dir).unwrap();
            db.put(b"durable", b"yes").unwrap();
            // dropped without an explicit flush -> only the WAL has this write
        }
        let db2 = LsmEngine::open(&dir).unwrap();
        assert_eq!(db2.get(b"durable").unwrap(), Some(b"yes".to_vec()));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mvcc_snapshot_read_across_flush() {
        let dir = tmp_dir("mvcc_flush");
        let mut db = LsmEngine::open(&dir).unwrap();
        let seq1 = db.put(b"k", b"v1").unwrap();
        db.flush().unwrap();
        let seq2 = db.put(b"k", b"v2").unwrap();
        assert_eq!(db.get_at(b"k", seq1).unwrap(), Some(b"v1".to_vec()));
        assert_eq!(db.get_at(b"k", seq2).unwrap(), Some(b"v2".to_vec()));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recovery_restores_correct_entry_count() {
        let dir = tmp_dir("entry_count_recovery");
        {
            let mut db = LsmEngine::open(&dir).unwrap();
            for i in 0..50u32 {
                db.put(format!("k{}", i).as_bytes(), b"v").unwrap();
            }
            db.flush().unwrap();
        }
        let db2 = LsmEngine::open(&dir).unwrap();
        assert!(db2.tiers.iter().all(|m| m.entry_count > 0));
        fs::remove_dir_all(&dir).ok();
    }
}
