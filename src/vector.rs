//! Flat (brute-force) vector index.
//!
//! Deliberately *not* HNSW/IVF: those trade recall and update complexity
//! for sub-linear search, and are the natural "next layer" on top of this
//! one (see docs/ARCHITECTURE.md). What this module focuses on instead is
//! the part that's easy to get wrong even for exact search — memory layout
//! and the inner distance loop:
//!
//! * Storage is struct-of-arrays: one flat `Vec<f32>` of length
//!   `ids.len() * dim`, not `Vec<Vec<f32>>`. A scan over N vectors touches
//!   one contiguous allocation instead of N pointer-chased ones — the
//!   difference between a linear prefetch pattern and a cache miss per row.
//! * Distance kernels use an 8-wide manual accumulator so LLVM can
//!   autovectorize them to AVX2 (8x f32 lanes) on stable Rust without
//!   `std::simd`. `cargo build --release` with `-C target-cpu=native`
//!   (or a `target-feature=+avx2` build) is where this actually pays off;
//!   see docs/PERF.md.
//! * Top-k selection is a bounded min-heap, O(n log k) instead of sorting
//!   all n candidates — the heap never holds more than k elements.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Higher (raw) dot product = more similar. Fine for pre-normalized
    /// embeddings; if vectors aren't unit-normed, prefer Cosine.
    Dot,
    /// Squared L2 distance, negated so "higher score = closer" holds for
    /// every metric uniformly at the call site.
    L2,
    Cosine,
}

#[derive(Debug)]
pub enum VectorError {
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::DimensionMismatch { expected, got } => {
                write!(f, "expected {}-dim vector, got {}", expected, got)
            }
        }
    }
}
impl std::error::Error for VectorError {}

pub struct VectorIndex {
    dim: usize,
    ids: Vec<u64>,
    data: Vec<f32>, // SoA: row i lives at data[i*dim .. (i+1)*dim]
}

impl VectorIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), VectorError> {
        if vector.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if let Some(pos) = self.ids.iter().position(|&x| x == id) {
            // overwrite in place — upsert semantics
            self.data[pos * self.dim..(pos + 1) * self.dim].copy_from_slice(vector);
        } else {
            self.ids.push(id);
            self.data.extend_from_slice(vector);
        }
        Ok(())
    }

    pub fn remove(&mut self, id: u64) -> bool {
        let Some(pos) = self.ids.iter().position(|&x| x == id) else {
            return false;
        };
        let last = self.ids.len() - 1;
        self.ids.swap(pos, last);
        self.ids.pop();
        // swap the row's floats with the last row's floats, then truncate
        for d in 0..self.dim {
            self.data.swap(pos * self.dim + d, last * self.dim + d);
        }
        self.data.truncate(last * self.dim);
        true
    }

    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Exact top-k nearest neighbors by brute-force scan. O(n * dim) for
    /// the scan, O(n log k) for selection.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: Metric,
    ) -> Result<Vec<(u64, f32)>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let query_norm = if metric == Metric::Cosine {
            norm(query)
        } else {
            1.0
        };
        let mut heap: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::with_capacity(k + 1);

        for i in 0..self.ids.len() {
            let row = self.row(i);
            let score = match metric {
                Metric::Dot => dot(query, row),
                Metric::L2 => -l2_sq(query, row),
                Metric::Cosine => {
                    let denom = query_norm * norm(row);
                    if denom == 0.0 {
                        0.0
                    } else {
                        dot(query, row) / denom
                    }
                }
            };
            let candidate = Scored {
                score,
                id: self.ids[i],
            };
            if heap.len() < k {
                heap.push(std::cmp::Reverse(candidate));
            } else if let Some(std::cmp::Reverse(worst)) = heap.peek() {
                if candidate.score > worst.score {
                    heap.pop();
                    heap.push(std::cmp::Reverse(candidate));
                }
            }
        }

        let mut out: Vec<(u64, f32)> = heap.into_iter().map(|r| (r.0.id, r.0.score)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(out)
    }
}

#[derive(Debug, Clone, Copy)]
struct Scored {
    score: f32,
    id: u64,
}
impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    // NaN → Less so it never wins in the bounded heap (unwrap_or(Equal)
    // would let NaN displace a real high-scoring result).
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Less)
    }
}

/// 8-lane manual accumulator. See module docs for why 8 and not "just call
/// `.iter().zip().map().sum()`" — the lane-parallel accumulation is what
/// breaks the serial dependency chain so LLVM can vectorize it.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        for l in 0..8 {
            acc[l] += a[base + l] * b[base + l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        for l in 0..8 {
            let d = a[base + l] - b[base + l];
            acc[l] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in (chunks * 8)..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[inline]
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_is_top_result() {
        let mut idx = VectorIndex::new(4);
        idx.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.insert(3, &[0.9, 0.1, 0.0, 0.0]).unwrap();
        let results = idx
            .search(&[1.0, 0.0, 0.0, 0.0], 2, Metric::Cosine)
            .unwrap();
        assert_eq!(results[0].0, 1);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn l2_prefers_nearest_point() {
        let mut idx = VectorIndex::new(2);
        idx.insert(1, &[0.0, 0.0]).unwrap();
        idx.insert(2, &[10.0, 10.0]).unwrap();
        idx.insert(3, &[0.5, 0.5]).unwrap();
        let results = idx.search(&[0.0, 0.0], 1, Metric::L2).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn remove_then_search_excludes_id() {
        let mut idx = VectorIndex::new(2);
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 1.0]).unwrap();
        idx.remove(1);
        let results = idx.search(&[1.0, 0.0], 5, Metric::Dot).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let mut idx = VectorIndex::new(3);
        let err = idx.insert(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            VectorError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn nan_score_does_not_displace_real_results() {
        // zero-vector query on Cosine -> NaN score; must not displace a real hit

        let mut idx = VectorIndex::new(2);
        idx.insert(1, &[1.0, 0.0]).unwrap();
        idx.insert(2, &[0.0, 0.0]).unwrap(); // zero vector -> NaN on Cosine
        let results = idx.search(&[1.0, 0.0], 2, Metric::Cosine).unwrap();
        // id=1 must be ranked first regardless of NaN from id=2
        assert_eq!(results[0].0, 1);
    }
}
