//! Reciprocal Rank Fusion. Combining raw vector cosine scores with BM25
//! scores directly doesn't work — the two live on incomparable scales — so
//! RRF fuses by *rank* instead of by score: `1 / (k + rank)` per list,
//! summed across lists. This is the standard approach most production
//! hybrid-search systems use instead of trying to calibrate score scales
//! against each other.

use std::cmp::Ordering;
use std::collections::HashMap;

/// `k_const` dampens the influence of rank 1 vs. rank 2 vs. rank 20 — 60 is
/// the value from the original RRF paper and is a reasonable default that
/// rarely needs tuning.
pub fn reciprocal_rank_fusion(result_lists: &[&[(u64, f32)]], k_const: f32) -> Vec<(u64, f32)> {
    let mut fused: HashMap<u64, f32> = HashMap::new();
    for list in result_lists {
        for (rank, (id, _score)) in list.iter().enumerate() {
            *fused.entry(*id).or_insert(0.0) += 1.0 / (k_const + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(u64, f32)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_ranked_high_in_both_lists_wins() {
        let vector_results: Vec<(u64, f32)> = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let text_results: Vec<(u64, f32)> = vec![(2, 5.0), (1, 4.0), (4, 3.0)];
        let fused = reciprocal_rank_fusion(&[&vector_results, &text_results], 60.0);
        // doc 1 is rank0+rank1, doc 2 is rank1+rank0 -> tied for first, both beat 3/4
        assert!(fused[0].0 == 1 || fused[0].0 == 2);
        assert!(fused.iter().position(|(id, _)| *id == 4).unwrap() > 1);
    }

    #[test]
    fn item_only_in_one_list_still_appears() {
        let vector_results: Vec<(u64, f32)> = vec![(1, 0.9)];
        let text_results: Vec<(u64, f32)> = vec![];
        let fused = reciprocal_rank_fusion(&[&vector_results, &text_results], 60.0);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].0, 1);
    }
}
