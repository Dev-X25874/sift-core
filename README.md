# sift-core

A hybrid search engine written in Rust. Combines a hand-rolled LSM storage engine with vector (ANN), full-text (BM25), and fusion (RRF) search — all in a single embeddable library with no external dependencies.

---

## What it does

- **LSM storage** — write-ahead log, memtable, SSTable flush and compaction, MVCC snapshot reads
- **Vector search** — brute-force ANN with Cosine, Dot, and L2 metrics; bounded top-k heap
- **Full-text search** — inverted index with Okapi BM25 ranking
- **Hybrid search** — Reciprocal Rank Fusion (RRF) over vector and text result lists
- **Crash safety** — WAL replayed on restart; atomic SSTable writes via `.tmp` + rename; fsync before WAL truncation

---

## Architecture

```
Engine
├── LsmEngine          — durable KV store (WAL + memtable + SSTables)
│   ├── MemTable       — BTreeMap, MVCC-keyed by (user_key, seq)
│   ├── SSTable        — sparse index + bloom filter + data block
│   ├── WAL            — FNV-1a checksummed append-only log
│   └── Compaction     — k-way merge, tombstone GC, atomic rename
├── VectorIndex        — flat SoA layout, O(n) scan, NaN-safe scoring
├── InvertedIndex      — posting lists, BM25, upsert-safe re-indexing
└── FusionSearch       — RRF over ranked result lists
```

---

## Getting started

**Requirements:** Rust 1.75+

```bash
git clone https://github.com/Dev-X25874/sift-core
cd sift-core
cargo build --release
cargo test
```

---

## Usage

```rust
use sift_core::Engine;

let mut engine = Engine::open("./data")?;

// Index a document
engine.upsert(
    1,
    &[0.1, 0.9, 0.3],       // vector embedding
    "rust systems programming database",  // text
    b"arbitrary blob payload",            // raw blob
)?;

// Hybrid search
let results = engine.query(
    &[0.1, 0.8, 0.4],
    "systems database",
    5,
)?;

for (id, score, blob) in results {
    println!("id={} score={:.4}", id, score);
}

// Delete
engine.delete(1)?;
```

---

## Storage format

**WAL record** (little-endian):
```
[checksum: u64][seq: u64][op: u8][key_len: u32][key][val_len: u32][val]
```

**SSTable layout** (written front-to-back, read back-to-front via footer):
```
[data block: sorted records]
[sparse index: (key, offset) every 16 records]
[bloom filter bits]
[footer: 7 × u64 — index_offset, index_count, bloom_offset,
                    bloom_num_bits, bloom_num_hashes, entry_count, magic]
```

Point lookup: bloom gate → binary search sparse index → one sequential block scan.

---

## Design decisions

**Why LSM over B-tree?** Write path is a sequential append to WAL + memtable insert. No random write amplification during ingestion, which matters when embedding vectors at write time.

**Why brute-force ANN?** An HNSW or IVF index adds significant implementation complexity for correctness gains that only matter past ~1M vectors. Brute-force with SIMD-friendly SoA layout is measurably fast up to that scale and trivially correct.

**Why single-tier compaction?** Leveled compaction (L0→Ln with size ratios) optimises write amplification at gigabyte scale. At this engine's target scale it would add complexity without changing the correctness or durability reasoning.

**Why FNV-1a for checksums and bloom?** Fast, dependency-free, and sufficient for corruption detection. Not cryptographic — not intended to be.

---

## Running benchmarks

```bash
cargo bench
```

Benchmarks cover vector search throughput, SSTable point-lookup latency, and LSM write throughput at varying flush thresholds. Results in `benches/`.

---

## Project layout

```
src/
  engine.rs       — public API, record encoding, hybrid query
  lsm.rs          — LSM engine (open, put, get, delete, flush, compact)
  memtable.rs     — in-memory sorted buffer
  sstable.rs      — on-disk sorted string table (writer + reader + iterator)
  wal.rs          — write-ahead log (append, replay, reset)
  compaction.rs   — k-way merge, lookup across tiers
  vector.rs       — vector index and ANN search
  text.rs         — inverted index and BM25 ranking
  fusion.rs       — reciprocal rank fusion
tests/
  integration.rs  — end-to-end Engine tests (restart, upsert, delete)
benches/
  bench.rs        — manual timing benchmarks
docs/
  ARCHITECTURE.md — deeper design notes and profiling guide
```

---

## License

MIT
