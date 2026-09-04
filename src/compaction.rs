//! Size-tiered compaction: merge N sstables into one, in a single sequential
//! pass, using a k-way merge over their already-sorted iterators.
//!
//! `InternalKey`'s Ord (user_key ASC, seq DESC) is exactly the merge order
//! we want, so the heap comparator does double duty: it produces the global
//! sort order *and* guarantees the newest version of a key surfaces first,
//! which is what makes the old-version GC below a single `if` instead of a
//! second pass.

use crate::sstable::{SSTable, SSTableIter, SSTableMeta, SSTableWriter};
use crate::{InternalKey, Value};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;
use std::path::Path;

/// Trigger policy: compact an L0-style tier once it accumulates this many
/// files. Real engines tune this against write amplification vs. read
/// amplification; a fixed threshold is the simplest thing that isn't wrong.
pub const COMPACTION_TRIGGER: usize = 4;

pub fn should_compact(file_count: usize) -> bool {
    file_count >= COMPACTION_TRIGGER
}

/// Merges `inputs` into a single new sstable at `output_path`.
///
/// `is_last_level`: when true, tombstones are dropped once they've done
/// their job of shadowing older versions in this same merge — there's
/// nothing further down to resurrect. When false (mid-tier compaction),
/// tombstones are preserved so an older version sitting in a
/// not-yet-compacted lower tier stays shadowed.
///
/// Merged output goes to `.tmp` first, then atomically renamed into place
/// after fsync. Inputs are only removed after a successful rename, so a
/// crash leaves the original files intact and a stale `.tmp` that gets
/// cleaned up on the next open().
pub fn compact(
    inputs: &[SSTableMeta],
    output_path: impl AsRef<Path>,
    is_last_level: bool,
) -> io::Result<SSTableMeta> {
    let output_path = output_path.as_ref();
    let tmp_path = output_path.with_extension("sst.tmp");

    let mut iters: Vec<SSTableIter> = inputs
        .iter()
        .map(|m| SSTableIter::open(&m.path))
        .collect::<io::Result<_>>()?;

    let mut heap: BinaryHeap<Reverse<(InternalKey, usize)>> = BinaryHeap::new();
    let mut buffered: Vec<Option<Value>> = vec![None; iters.len()];

    for (i, it) in iters.iter_mut().enumerate() {
        if let Some(item) = it.next() {
            let (ik, v) = item?;
            heap.push(Reverse((ik, i)));
            buffered[i] = Some(v);
        }
    }

    let expected_entries: u64 = inputs.iter().map(|m| m.entry_count).sum();
    let mut writer = SSTableWriter::create(&tmp_path, expected_entries as usize)?;
    let mut last_emitted_key: Option<Vec<u8>> = None;

    while let Some(Reverse((ik, idx))) = heap.pop() {
        let value = buffered[idx]
            .take()
            .expect("heap entry without buffered value");

        if let Some(item) = iters[idx].next() {
            let (next_ik, next_val) = item?;
            heap.push(Reverse((next_ik, idx)));
            buffered[idx] = Some(next_val);
        }

        let is_newest_for_key = last_emitted_key.as_deref() != Some(ik.user_key.as_slice());
        if !is_newest_for_key {
            continue; // shadowed by a newer version already emitted -> GC'd
        }
        last_emitted_key = Some(ik.user_key.clone());

        if is_last_level && matches!(value, Value::Delete) {
            continue; // tombstone has nothing left below it to shadow -> GC'd
        }

        writer.add(&ik, &value)?;
    }

    // finish() fsyncs the tmp file to durable storage before returning.
    let mut meta = writer.finish()?;

    // Atomic rename: the merged file becomes visible at its final path
    // only after it is fully written and durable. If the rename fails
    // we leave the .tmp file behind (harmless; no inputs have been touched).
    std::fs::rename(&tmp_path, output_path)?;
    meta.path = output_path.to_path_buf();

    Ok(meta)
}

/// Convenience used by the engine to decide whether a given `key` is truly
/// absent (vs. just not-yet-flushed) by consulting on-disk tiers in
/// newest-to-oldest order, stopping at the first definitive answer.
pub fn lookup_in_tiers(
    tiers: &[SSTableMeta],
    key: &[u8],
    as_of: crate::Seq,
) -> io::Result<Option<Value>> {
    for meta in tiers {
        let sst = SSTable::open(&meta.path)?;
        if let Some(v) = sst.get(key, as_of)? {
            return Ok(Some(v));
        }
    }
    Ok(None)
}
