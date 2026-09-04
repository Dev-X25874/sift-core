# Architecture

sift-core is a hybrid (vector + full-text) search engine built on a from-scratch
LSM storage engine. It exists to be read, not to be run at scale — every
module is small enough to hold in your head, and every non-obvious decision
is a comment next to the code it explains, not a slide deck somewhere else.

```
                    ┌─────────────────────┐
                    │       Engine         │  upsert / delete / query
                    └──────────┬───────────┘
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
      │ VectorIndex    │ │ InvertedIndex│ │  LsmEngine    │
      │ (flat, SoA)    │ │ (BM25)       │ │  (doc blobs)  │
      └───────────────┘ └──────────────┘ └──────┬────────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                        ▼                       ▼
                    ┌──────────┐            ┌──────────────┐       ┌─────────────┐
                    │   WAL     │            │  MemTable     │       │  SSTable(s)  │
                    │ (durability)│          │ (sorted MVCC) │       │ (immutable)  │
                    └──────────┘            └──────────────┘       └──────┬───────┘
                                                                            │
                                                                     ┌──────▼──────┐
                                                                     │ Compaction   │
                                                                     │ (k-way merge)│
                                                                     └─────────────┘
```

Query path: `Engine::query` runs the vector index and the inverted index
independently, widens each to a candidate pool bigger than `k` (see
`CANDIDATE_POOL_MULTIPLIER` in `engine.rs`), fuses the two ranked lists with
Reciprocal Rank Fusion, then fetches blobs for the top `k` fused ids from the
LSM store.

## Storage engine

* **WAL** (`src/wal.rs`) — length-prefixed records with an FNV-1a checksum.
  Replay stops at the first bad checksum or short read instead of erroring,
  because a torn tail record is the *expected* shape of a crash mid-write,
  not a corruption event to panic over.
* **MemTable** (`src/memtable.rs`) — `BTreeMap<InternalKey, Value>`.
  `InternalKey` sorts `(user_key ASC, seq DESC)`, which makes MVCC point
  lookups a single forward range-scan instead of a secondary version index.
* **SSTable** (`src/sstable.rs`) — data block + sparse index + bloom filter
  + footer, read back-to-front from EOF. A lookup is: bloom filter
  (probably-absent → skip the file) → binary search the in-memory sparse
  index → one seek + sequential scan of one block.
* **Compaction** (`src/compaction.rs`) — k-way merge over sstable iterators
  using a min-heap keyed by `InternalKey`'s own `Ord`. Because that `Ord`
  already sorts newest-version-first within a key, GC of shadowed versions
  and (at the bottom level) tombstones falls out as a single `if`, not a
  second pass.
* **LsmEngine** (`src/lsm.rs`) — wires the above into `put` / `delete` /
  `get_at(key, snapshot_seq)`. Single-tier compaction (not LevelDB-style
  `L0..Ln` leveling) — see "Explicit non-goals" below.

## Search layer

* **VectorIndex** (`src/vector.rs`) — flat brute-force search, struct-of-
  arrays layout (one contiguous `Vec<f32>`, not `Vec<Vec<f32>>`), 8-wide
  manual accumulator in the distance kernels so LLVM can autovectorize on
  stable Rust, bounded min-heap for O(n log k) top-k selection.
* **InvertedIndex** (`src/text.rs`) — postings lists + Okapi BM25.
* **fusion** (`src/fusion.rs`) — Reciprocal Rank Fusion across the two
  ranked lists; score scales from cosine similarity and BM25 aren't
  comparable, so fusion happens over rank, not raw score.

## Explicit non-goals (and what replacing them would look like)

This is a "systems primitives" portfolio piece, not a distributed database,
and it's more useful to say precisely what's missing than to imply it's all
here:

* **No ANN index.** Vector search is exact brute-force, O(n·dim) per query.
  The natural next layer is an HNSW graph or IVF partitioning on top of the
  same SoA storage — the interesting engineering there is mostly in the
  index-build and incremental-update path, not the distance kernel.
* **No leveled compaction.** One tier, fully re-merged on every compaction.
  Fine at the data sizes this repo is meant to be read/run at; wrong past
  that, because it makes compaction cost `O(total data)` instead of
  `O(size of the level being compacted)`.
* **No replication, no distributed anything.** No leader election, no
  Raft/Paxos log, no leases, no chaos testing. This is a single-node engine.
  Turning it into a distributed one is a different (and larger) project;
  wiring the LSM's WAL into a replicated log is the natural seam.
* **No block compression** in sstables, no two-level (index-of-indexes)
  block index — both matter once indexes stop fitting comfortably in
  memory.
* **No real tokenizer.** Text indexing is lowercase + split-on-non-
  alphanumeric. No stemming, no stopwords, no phrase/positional queries.
* **No concurrency.** Every structure here is `&mut self` — there's no
  reader-writer story, no lock-free memtable, no background compaction
  thread. Single-threaded by design, so the storage-engine logic isn't
  tangled up with synchronization logic.

See `docs/PERF.md` for how to actually go measure where time goes instead
of guessing from this list.
