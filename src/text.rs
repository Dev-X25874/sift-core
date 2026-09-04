//! Inverted index with BM25 ranking. No stemming, no stopword list, no
//! phrase queries — minimum viable full-text layer to demonstrate
//! postings-list structure and BM25 math. Tokenizer/analyzer pipeline
//! improvements are future work (see docs/ARCHITECTURE.md).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Debug, Clone)]
struct Posting {
    doc_id: u64,
    term_freq: u32,
}

pub struct InvertedIndex {
    postings: HashMap<String, Vec<Posting>>,
    doc_lengths: HashMap<u64, u32>,
    total_doc_length: u64,
    doc_count: u64,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_doc_length: 0,
            doc_count: 0,
        }
    }

    /// Upsert: re-indexing a `doc_id` replaces its prior terms.
    pub fn index_document(&mut self, doc_id: u64, text: &str) {
        self.remove_document(doc_id);

        let tokens = tokenize(text);
        let len = tokens.len() as u32;
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *term_freqs.entry(t).or_insert(0) += 1;
        }
        for (term, freq) in term_freqs {
            self.postings.entry(term).or_default().push(Posting {
                doc_id,
                term_freq: freq,
            });
        }
        self.doc_lengths.insert(doc_id, len);
        self.total_doc_length += len as u64;
        self.doc_count += 1;
    }

    pub fn remove_document(&mut self, doc_id: u64) {
        if let Some(len) = self.doc_lengths.remove(&doc_id) {
            self.total_doc_length = self.total_doc_length.saturating_sub(len as u64);
            self.doc_count = self.doc_count.saturating_sub(1);
            for postings in self.postings.values_mut() {
                postings.retain(|p| p.doc_id != doc_id);
            }
            // Drop empty posting list entries — otherwise high-churn workloads
            // accumulate empty Vecs in the HashMap indefinitely.
            self.postings.retain(|_, v| !v.is_empty());
        }
    }

    fn avg_doc_length(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.total_doc_length as f32 / self.doc_count as f32
        }
    }

    /// Okapi BM25 over the union of postings for each distinct query term.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        if self.doc_count == 0 {
            return Vec::new();
        }
        let avgdl = self.avg_doc_length().max(1.0);
        let n = self.doc_count as f32;

        let mut scores: HashMap<u64, f32> = HashMap::new();
        let mut seen_terms: HashSet<String> = HashSet::new();

        for term in tokenize(query) {
            if !seen_terms.insert(term.clone()) {
                continue; // duplicate query term counts once per doc-freq IDF
            }
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for p in postings {
                let dl = *self.doc_lengths.get(&p.doc_id).unwrap_or(&0) as f32;
                let tf = p.term_freq as f32;
                let denom = tf + K1 * (1.0 - B + B * (dl / avgdl));
                let score = idf * (tf * (K1 + 1.0)) / denom;
                *scores.entry(p.doc_id).or_insert(0.0) += score;
            }
        }

        let mut out: Vec<(u64, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_term_match_outranks_unrelated_doc() {
        let mut idx = InvertedIndex::new();
        idx.index_document(
            1,
            "sift-core is a hybrid search engine for vectors and text",
        );
        idx.index_document(2, "the weather today is mild and pleasant");
        let results = idx.search("search engine vectors", 5);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn removing_a_document_drops_it_from_results() {
        let mut idx = InvertedIndex::new();
        idx.index_document(1, "rust systems programming");
        idx.index_document(2, "rust systems programming and databases");
        idx.remove_document(2);
        let results = idx.search("rust systems", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn reindexing_a_doc_id_replaces_its_terms() {
        let mut idx = InvertedIndex::new();
        idx.index_document(1, "apples and oranges");
        idx.index_document(1, "bananas only");
        assert!(idx.search("apples", 5).is_empty());
        assert_eq!(idx.search("bananas", 5)[0].0, 1);
    }

    #[test]
    fn remove_document_prunes_empty_posting_lists() {
        let mut idx = InvertedIndex::new();
        idx.index_document(1, "unique term alpha");
        idx.remove_document(1);
        // posting list for "unique", "term", "alpha" should be gone entirely
        assert!(idx.postings.is_empty());
    }
}
