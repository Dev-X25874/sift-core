//! Manual `Instant`-based benchmarks. No `criterion` dependency on purpose —
//! this repo is zero-dependency by design (see README), and for
//! order-of-magnitude perf sanity checks a warm-up loop + median of a few
//! runs is plenty. For real regression tracking, `criterion` (or
//! `hyperfine` around `--release` runs of this binary) is the right call.
//!
//! Run with: `cargo run --release --bin sift-bench`

use sift_core::vector::{Metric, VectorIndex};
use std::time::Instant;

/// xorshift64 — deterministic, dependency-free PRNG. Good enough for
/// synthetic benchmark data; not for anything security-sensitive.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32) // uniform in [0, 1)
    }
}

fn bench_vector_search(n: usize, dim: usize, k: usize, queries: usize) {
    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let mut index = VectorIndex::new(dim);

    for id in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        index.insert(id as u64, &v).unwrap();
    }

    let query_vecs: Vec<Vec<f32>> = (0..queries)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect();

    // warm-up
    for q in &query_vecs {
        let _ = index.search(q, k, Metric::Cosine).unwrap();
    }

    let start = Instant::now();
    for q in &query_vecs {
        let _ = index.search(q, k, Metric::Cosine).unwrap();
    }
    let elapsed = start.elapsed();

    println!(
        "vector search: n={:<7} dim={:<4} k={:<3} -> {:>8.3} ms/query  ({:.1} queries/sec)",
        n,
        dim,
        k,
        elapsed.as_secs_f64() * 1000.0 / queries as f64,
        queries as f64 / elapsed.as_secs_f64()
    );
}

fn bench_lsm_writes(n: usize) {
    let dir = std::env::temp_dir().join(format!("sift_core_bench_lsm_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let mut db = sift_core::lsm::LsmEngine::open(&dir).unwrap();

    let start = Instant::now();
    for i in 0..n {
        let key = format!("bench-key-{:08}", i);
        let val = format!("bench-value-{}", i);
        db.put(key.as_bytes(), val.as_bytes()).unwrap();
    }
    let elapsed = start.elapsed();

    println!(
        "lsm sequential put (fsync'd): n={:<7} -> {:>8.3} us/write  ({:.0} writes/sec)",
        n,
        elapsed.as_secs_f64() * 1_000_000.0 / n as f64,
        n as f64 / elapsed.as_secs_f64()
    );

    std::fs::remove_dir_all(&dir).ok();
}

fn main() {
    println!(
        "sift-core micro-benchmarks (debug-vs-release matters a lot here — run with --release)\n"
    );

    bench_vector_search(10_000, 128, 10, 200);
    bench_vector_search(100_000, 128, 10, 50);
    bench_vector_search(10_000, 768, 10, 200);

    bench_lsm_writes(2_000);
}
