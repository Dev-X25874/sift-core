// sift-core: a small hybrid (vector + full-text) search engine built on a
//! from-scratch LSM storage engine. Written to be read end-to-end in an
//! afternoon — every layer is <500 LOC and has no external dependencies.
//!
//! Layers (bottom to top):
//!   wal        -> crash-durable append log
//!   memtable   -> in-memory sorted MVCC buffer
//!   sstable    -> immutable on-disk sorted run + sparse index + bloom filter
//!   compaction -> k-way merge of sstables, GC of old versions/tombstones
//!   lsm        -> ties the above into a Bitcask/LevelDB-shaped KV engine
//!   vector     -> flat SoA vector index, top-k via bounded heap
//!   text       -> inverted index + BM25
//!   fusion     -> reciprocal rank fusion across vector/text result sets
//!   engine     -> the public hybrid-search API

pub mod compaction;
pub mod engine;
pub mod fusion;
pub mod lsm;
pub mod memtable;
pub mod sstable;
pub mod text;
pub mod vector;
pub mod wal;

pub use engine::{Engine, EngineError};

/// A logical write timestamp. Every mutation to the LSM gets a strictly
/// increasing seq; reads pin a seq to get snapshot isolation (MVCC).
pub type Seq = u64;

/// Internal ordering key: (user_key ASC, seq DESC). Sorting seq descending
/// within a user key means "newest version first" during a forward scan,
/// which is exactly what point lookups and compaction GC want.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InternalKey {
    pub user_key: Vec<u8>,
    pub seq: Seq,
}

impl InternalKey {
    pub fn new(user_key: impl Into<Vec<u8>>, seq: Seq) -> Self {
        Self {
            user_key: user_key.into(),
            seq,
        }
    }
}

impl Ord for InternalKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.user_key
            .cmp(&other.user_key)
            .then_with(|| other.seq.cmp(&self.seq)) // seq DESC within a key
    }
}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A value is either live bytes or a tombstone (delete marker). Tombstones
/// have to survive in the LSM until every older version is compacted away,
/// otherwise a delete can "resurrect" a stale value from an older run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Put(Vec<u8>),
    Delete,
}

impl Value {
    pub fn as_put(&self) -> Option<&[u8]> {
        match self {
            Value::Put(v) => Some(v),
            Value::Delete => None,
        }
    }
}
