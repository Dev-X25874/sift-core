//! On-disk sorted string table.
//!
//! Layout, back to front (we write forward, but readers seek from EOF):
//!
//!   [ data block: sorted records ]
//!   [ sparse index: one (key, offset) every SPARSE_INTERVAL records ]
//!   [ bloom filter bits ]
//!   [ fixed-size footer ]
//!
//! Point lookup: bloom filter (probably-not-present → skip entirely) →
//! binary search sparse index → one sequential read of that block.
//! Same shape as LevelDB/RocksDB, minus block compression and two-level
//! index — first things you'd add when the dataset outgrows memory.

use crate::{InternalKey, Seq, Value};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const SPARSE_INTERVAL: usize = 16;
const FOOTER_MAGIC: u64 = 0x7470_7566_6665_7221; // "tpuffer!"

// Footer layout (7 × u64 = 56 bytes):
//   [index_offset][index_count][bloom_offset][bloom_num_bits][bloom_num_hashes][entry_count][magic]
const FOOTER_SIZE: usize = 8 * 7;

const BITS_PER_KEY: usize = 10;

// ---------- bloom filter ----------

struct BloomFilter {
    bits: Vec<u8>,
    num_bits: u64,
    num_hashes: u32,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

impl BloomFilter {
    fn with_capacity(expected_entries: usize) -> Self {
        let num_bits = (expected_entries.max(1) * BITS_PER_KEY).max(64) as u64;
        let num_bytes = (num_bits as usize).div_ceil(8);
        let num_hashes = ((BITS_PER_KEY as f64) * std::f64::consts::LN_2)
            .round()
            .clamp(1.0, 30.0) as u32;
        Self {
            bits: vec![0u8; num_bytes],
            num_bits,
            num_hashes,
        }
    }

    /// Double-hashing without allocation: h2 is FNV-1a over [h1_bytes... || 0x01].
    /// Avoids a Vec alloc on every probe — this is in the hot path for both
    /// reads and writes.
    fn hashes(&self, key: &[u8]) -> (u64, u64) {
        let h1 = fnv1a(key);
        // Extend the FNV state from h1 with a single suffix byte instead of
        // copying key into a new Vec.
        const PRIME: u64 = 0x100000001b3;
        let h2 = (h1 ^ 0x01u64).wrapping_mul(PRIME);
        (h1, h2)
    }

    fn add(&mut self, key: &[u8]) {
        let (h1, h2) = self.hashes(key);
        for i in 0..self.num_hashes as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.num_bits;
            self.bits[(bit / 8) as usize] |= 1 << (bit % 8);
        }
    }

    fn might_contain(&self, key: &[u8]) -> bool {
        let (h1, h2) = self.hashes(key);
        for i in 0..self.num_hashes as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.num_bits;
            if self.bits[(bit / 8) as usize] & (1 << (bit % 8)) == 0 {
                return false;
            }
        }
        true
    }

    fn from_raw(bits: Vec<u8>, num_bits: u64, num_hashes: u32) -> Self {
        Self {
            bits,
            num_bits,
            num_hashes,
        }
    }
}

// ---------- writer ----------

pub struct SSTableMeta {
    pub path: PathBuf,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub entry_count: u64,
}

pub struct SSTableWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    offset: u64,
    sparse_index: Vec<(Vec<u8>, u64)>,
    bloom: BloomFilter,
    entries_since_index: usize,
    entry_count: u64,
    min_key: Option<Vec<u8>>,
    max_key: Vec<u8>,
}

impl SSTableWriter {
    pub fn create(path: impl AsRef<Path>, expected_entries: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            offset: 0,
            sparse_index: Vec::new(),
            bloom: BloomFilter::with_capacity(expected_entries),
            entries_since_index: 0,
            entry_count: 0,
            min_key: None,
            max_key: Vec::new(),
        })
    }

    /// Caller MUST supply entries in InternalKey order (user_key ASC, seq DESC).
    /// Writer doesn't sort — same contract as LevelDB/RocksDB.
    pub fn add(&mut self, ik: &InternalKey, value: &Value) -> io::Result<()> {
        if self.entries_since_index == 0 {
            self.sparse_index.push((ik.user_key.clone(), self.offset));
        }
        if self.min_key.is_none() {
            self.min_key = Some(ik.user_key.clone());
        }
        self.max_key = ik.user_key.clone();
        self.bloom.add(&ik.user_key);

        let (op, val_bytes): (u8, &[u8]) = match value {
            Value::Put(v) => (0, v.as_slice()),
            Value::Delete => (1, &[]),
        };

        let mut record = Vec::with_capacity(4 + ik.user_key.len() + 8 + 1 + 4 + val_bytes.len());
        record.extend_from_slice(&(ik.user_key.len() as u32).to_le_bytes());
        record.extend_from_slice(&ik.user_key);
        record.extend_from_slice(&ik.seq.to_le_bytes());
        record.push(op);
        record.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
        record.extend_from_slice(val_bytes);

        self.writer.write_all(&record)?;
        self.offset += record.len() as u64;
        self.entry_count += 1;
        self.entries_since_index = (self.entries_since_index + 1) % SPARSE_INTERVAL;
        Ok(())
    }

    /// Flushes and fsyncs the SSTable to disk. The WAL is truncated right
    /// after this returns, so the SSTable must be durable first.
    pub fn finish(mut self) -> io::Result<SSTableMeta> {
        let index_offset = self.offset;
        let mut index_buf = Vec::new();
        for (key, off) in &self.sparse_index {
            index_buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            index_buf.extend_from_slice(key);
            index_buf.extend_from_slice(&off.to_le_bytes());
        }
        self.writer.write_all(&index_buf)?;
        self.offset += index_buf.len() as u64;

        let bloom_offset = self.offset;
        self.writer.write_all(&self.bloom.bits)?;
        self.offset += self.bloom.bits.len() as u64;

        // entry_count written into footer so LsmEngine::open() can
        // reconstruct SSTableMeta faithfully after a restart.
        let mut footer = Vec::with_capacity(FOOTER_SIZE);
        footer.extend_from_slice(&index_offset.to_le_bytes());
        footer.extend_from_slice(&(self.sparse_index.len() as u64).to_le_bytes());
        footer.extend_from_slice(&bloom_offset.to_le_bytes());
        footer.extend_from_slice(&self.bloom.num_bits.to_le_bytes());
        footer.extend_from_slice(&(self.bloom.num_hashes as u64).to_le_bytes());
        footer.extend_from_slice(&self.entry_count.to_le_bytes());
        footer.extend_from_slice(&FOOTER_MAGIC.to_le_bytes());
        self.writer.write_all(&footer)?;

        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;

        Ok(SSTableMeta {
            path: self.path,
            min_key: self.min_key.unwrap_or_default(),
            max_key: self.max_key,
            entry_count: self.entry_count,
        })
    }
}

// ---------- reader ----------

pub struct SSTable {
    path: PathBuf,
    index: Vec<(Vec<u8>, u64)>,
    bloom: BloomFilter,
    data_end: u64,    // == index_offset; records live in [0, data_end)
    entry_count: u64, // read from footer, used to size bloom on compaction
}

impl SSTable {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable too small",
            ));
        }

        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
        let mut footer = [0u8; FOOTER_SIZE];
        file.read_exact(&mut footer)?;
        let index_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
        let index_count = u64::from_le_bytes(footer[8..16].try_into().unwrap());
        let bloom_offset = u64::from_le_bytes(footer[16..24].try_into().unwrap());
        let bloom_num_bits = u64::from_le_bytes(footer[24..32].try_into().unwrap());
        let bloom_num_hashes = u64::from_le_bytes(footer[32..40].try_into().unwrap());
        let entry_count = u64::from_le_bytes(footer[40..48].try_into().unwrap());
        let magic = u64::from_le_bytes(footer[48..56].try_into().unwrap());
        if magic != FOOTER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad sstable footer magic",
            ));
        }

        // Bounds-check footer offsets before using them in arithmetic.
        // Unchecked subtraction here wraps to a huge u64 on corrupt input,
        // causing a multi-GB allocation followed by an OOM or panic.
        let data_end_bound = file_len - FOOTER_SIZE as u64;
        if index_offset > data_end_bound
            || bloom_offset > data_end_bound
            || bloom_offset < index_offset
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable footer offsets out of range",
            ));
        }

        // read index block
        file.seek(SeekFrom::Start(index_offset))?;
        let index_len = bloom_offset - index_offset;
        let mut index_buf = vec![0u8; index_len as usize];
        file.read_exact(&mut index_buf)?;
        let mut index = Vec::with_capacity(index_count as usize);
        let mut cursor = 0usize;
        while cursor < index_buf.len() {
            let klen =
                u32::from_le_bytes(index_buf[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let key = index_buf[cursor..cursor + klen].to_vec();
            cursor += klen;
            let off = u64::from_le_bytes(index_buf[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            index.push((key, off));
        }

        // read bloom block
        let bloom_len = data_end_bound - bloom_offset;
        file.seek(SeekFrom::Start(bloom_offset))?;
        let mut bloom_bits = vec![0u8; bloom_len as usize];
        file.read_exact(&mut bloom_bits)?;
        let bloom = BloomFilter::from_raw(bloom_bits, bloom_num_bits, bloom_num_hashes as u32);

        Ok(Self {
            path,
            index,
            bloom,
            data_end: index_offset,
            entry_count,
        })
    }

    /// Reconstruct an SSTableMeta from an on-disk file. Used by
    /// LsmEngine::open() to restore accurate entry_count and min_key after a
    /// restart. max_key is not stored in the footer; leave it empty — nothing
    /// in the current read or compaction path checks it.
    pub fn into_meta(self) -> SSTableMeta {
        let min_key = self
            .index
            .first()
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        SSTableMeta {
            path: self.path,
            min_key,
            max_key: Vec::new(),
            entry_count: self.entry_count,
        }
    }

    /// Binary search the sparse index for the block that could contain `key`.
    ///
    /// When a user_key has >= SPARSE_INTERVAL versions, multiple consecutive
    /// sparse index entries share the same user_key. Walk left to the first
    /// occurrence so we don't skip newer (higher-seq) versions that appear
    /// earlier in the file.
    fn block_start_offset(&self, key: &[u8]) -> u64 {
        // If the key is in the sparse index at position i, earlier versions
        // (higher seq) may live in the previous block. Go back one block so
        // we don't skip them. The scan stops itself via the Greater branch.
        let i = self
            .index
            .binary_search_by(|(k, _)| k.as_slice().cmp(key))
            .unwrap_or_else(|i| i);
        if i > 0 {
            self.index[i - 1].1
        } else {
            0
        }
    }

    pub fn get(&self, key: &[u8], as_of: Seq) -> io::Result<Option<Value>> {
        if !self.bloom.might_contain(key) {
            return Ok(None);
        }
        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut offset = self.block_start_offset(key);

        loop {
            if offset >= self.data_end {
                return Ok(None);
            }
            reader.seek(SeekFrom::Start(offset))?;
            let rec = match read_record(&mut reader)? {
                Some(r) => r,
                None => return Ok(None),
            };
            offset += rec.encoded_len as u64;

            match rec.user_key.as_slice().cmp(key) {
                std::cmp::Ordering::Greater => return Ok(None), // sorted → not present
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => {
                    if rec.seq <= as_of {
                        return Ok(Some(rec.value));
                    }
                    continue; // newer than snapshot; scan older versions
                }
            }
        }
    }

    pub fn min_max(&self) -> (Option<&[u8]>, Option<&[u8]>) {
        (self.index.first().map(|(k, _)| k.as_slice()), None)
    }
}

struct RawRecord {
    user_key: Vec<u8>,
    seq: Seq,
    value: Value,
    encoded_len: usize,
}

fn read_record<R: Read>(reader: &mut R) -> io::Result<Option<RawRecord>> {
    let mut klen_buf = [0u8; 4];
    if reader.read_exact(&mut klen_buf).is_err() {
        return Ok(None);
    }
    let klen = u32::from_le_bytes(klen_buf) as usize;
    let mut key = vec![0u8; klen];
    reader.read_exact(&mut key)?;
    let mut seq_buf = [0u8; 8];
    reader.read_exact(&mut seq_buf)?;
    let mut op_buf = [0u8; 1];
    reader.read_exact(&mut op_buf)?;
    let mut vlen_buf = [0u8; 4];
    reader.read_exact(&mut vlen_buf)?;
    let vlen = u32::from_le_bytes(vlen_buf) as usize;
    let mut val = vec![0u8; vlen];
    reader.read_exact(&mut val)?;

    let encoded_len = 4 + klen + 8 + 1 + 4 + vlen;
    let seq = u64::from_le_bytes(seq_buf);
    let value = if op_buf[0] == 0 {
        Value::Put(val)
    } else {
        Value::Delete
    };
    Ok(Some(RawRecord {
        user_key: key,
        seq,
        value,
        encoded_len,
    }))
}

/// Sequential full-scan iterator over a data block — used by compaction's k-way merge.
/// Point lookups use `SSTable::get` instead (index + seek).
pub struct SSTableIter {
    reader: BufReader<File>,
    remaining: u64,
}

impl SSTableIter {
    /// Opens the file once and reads `data_end` directly from the footer,
    /// rather than constructing a full SSTable and then re-opening the file.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable too small",
            ));
        }

        // Read just the index_offset (first field) from the footer —
        // that's all we need; everything else is for point lookups.
        file.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
        let mut footer_buf = [0u8; FOOTER_SIZE];
        file.read_exact(&mut footer_buf)?;
        let index_offset = u64::from_le_bytes(footer_buf[0..8].try_into().unwrap());
        let magic = u64::from_le_bytes(footer_buf[48..56].try_into().unwrap());
        if magic != FOOTER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad sstable footer magic",
            ));
        }

        // Rewind to start for the sequential scan.
        file.seek(SeekFrom::Start(0))?;
        Ok(Self {
            reader: BufReader::new(file),
            remaining: index_offset,
        })
    }
}

impl Iterator for SSTableIter {
    type Item = io::Result<(InternalKey, Value)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match read_record(&mut self.reader) {
            Ok(Some(rec)) => {
                self.remaining = self.remaining.saturating_sub(rec.encoded_len as u64);
                Some(Ok((InternalKey::new(rec.user_key, rec.seq), rec.value)))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!(
            "sift_core_sst_test_{}_{}",
            std::process::id(),
            name
        ));
        p
    }

    #[test]
    fn write_then_point_lookups() {
        let path = tmp_path("basic.sst");
        let mut w = SSTableWriter::create(&path, 100).unwrap();
        for i in 0..100u32 {
            let key = format!("key{:04}", i).into_bytes();
            let val = format!("val{}", i).into_bytes();
            w.add(&InternalKey::new(key, 1), &Value::Put(val)).unwrap();
        }
        w.finish().unwrap();

        let sst = SSTable::open(&path).unwrap();
        assert_eq!(
            sst.get(b"key0050", 10).unwrap(),
            Some(Value::Put(b"val50".to_vec()))
        );
        assert_eq!(
            sst.get(b"key0099", 10).unwrap(),
            Some(Value::Put(b"val99".to_vec()))
        );
        assert_eq!(sst.get(b"nope", 10).unwrap(), None);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn entry_count_survives_round_trip() {
        let path = tmp_path("entry_count.sst");
        let mut w = SSTableWriter::create(&path, 50).unwrap();
        for i in 0..50u32 {
            let key = format!("k{:04}", i).into_bytes();
            w.add(&InternalKey::new(key, 1), &Value::Put(b"v".to_vec()))
                .unwrap();
        }
        w.finish().unwrap();

        let meta = SSTable::open(&path).unwrap().into_meta();
        assert_eq!(meta.entry_count, 50);
        std::fs::remove_file(&path).ok();
    }

    /// A key with >= SPARSE_INTERVAL versions must still be found at the
    /// correct (newest <= as_of) version via point lookup.
    #[test]
    fn many_versions_of_same_key_point_lookup() {
        let path = tmp_path("multiversion.sst");
        let mut w = SSTableWriter::create(&path, 22).unwrap();
        w.add(
            &InternalKey::new(b"aaa".to_vec(), 1),
            &Value::Put(b"anchor-low".to_vec()),
        )
        .unwrap();
        for seq in (1u64..=20).rev() {
            let val = format!("hot-v{}", seq).into_bytes();
            w.add(&InternalKey::new(b"hot".to_vec(), seq), &Value::Put(val))
                .unwrap();
        }
        w.add(
            &InternalKey::new(b"zzz".to_vec(), 1),
            &Value::Put(b"anchor-high".to_vec()),
        )
        .unwrap();
        w.finish().unwrap();

        let sst = SSTable::open(&path).unwrap();
        assert_eq!(
            sst.get(b"hot", Seq::MAX).unwrap(),
            Some(Value::Put(b"hot-v20".to_vec()))
        );
        assert_eq!(
            sst.get(b"hot", 5).unwrap(),
            Some(Value::Put(b"hot-v5".to_vec()))
        );
        assert_eq!(sst.get(b"hot", 0).unwrap(), None);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bloom_filter_has_no_false_negatives() {
        let mut bf = BloomFilter::with_capacity(1000);
        let keys: Vec<Vec<u8>> = (0..1000).map(|i| format!("k{}", i).into_bytes()).collect();
        for k in &keys {
            bf.add(k);
        }
        for k in &keys {
            assert!(bf.might_contain(k));
        }
    }
}
