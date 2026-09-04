//! The public surface: a namespace that stores a blob, a vector, and text
//! per document id, and answers hybrid (vector + BM25, fused via RRF)
//! queries. Everything below this file is a reusable layer; this file is
//! just wiring.

use crate::fusion;
use crate::lsm::LsmEngine;
use crate::text::InvertedIndex;
use crate::vector::{Metric, VectorError, VectorIndex};
use std::fmt;
use std::io;
use std::path::Path;

#[derive(Debug)]
pub enum EngineError {
    Io(io::Error),
    Vector(VectorError),
    Corrupt(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Io(e) => write!(f, "io error: {}", e),
            EngineError::Vector(e) => write!(f, "vector error: {}", e),
            EngineError::Corrupt(msg) => write!(f, "corrupt document record: {}", msg),
        }
    }
}
impl std::error::Error for EngineError {}
impl From<io::Error> for EngineError {
    fn from(e: io::Error) -> Self {
        EngineError::Io(e)
    }
}
impl From<VectorError> for EngineError {
    fn from(e: VectorError) -> Self {
        EngineError::Vector(e)
    }
}

pub struct SearchHit {
    pub id: u64,
    pub score: f32,
    pub blob: Vec<u8>,
}

/// How many candidates each sub-index contributes to fusion before
/// truncating to the caller's requested `k`. Widening the candidate pool
/// past `k` before fusing is standard hybrid-search practice: a doc that's
/// (say) rank 15 in both vector and text search can out-rank one that's
/// rank 1 in only one of them, but only if both lists are wide enough to
/// contain it in the first place.
const CANDIDATE_POOL_MULTIPLIER: usize = 4;
const MIN_CANDIDATE_POOL: usize = 20;

/// Wire format for a stored document: `[vec_len:u32][vec f32 LE...][text_len:u32][text utf8][blob_len:u32][blob]`.
/// Deliberately not the WAL's or sstable's record format — this is a level
/// up, the *value* the LSM stores, opaque to everything below `engine.rs`.
fn encode_record(vector: &[f32], text: &str, blob: &[u8]) -> Vec<u8> {
    let text_bytes = text.as_bytes();
    let mut buf = Vec::with_capacity(4 + vector.len() * 4 + 4 + text_bytes.len() + 4 + blob.len());
    buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
    for f in vector {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(text_bytes);
    buf.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    buf.extend_from_slice(blob);
    buf
}

fn decode_record(bytes: &[u8]) -> Result<(Vec<f32>, String, Vec<u8>), EngineError> {
    let mut c = 0usize;
    let read_u32 = |bytes: &[u8], c: &mut usize| -> Result<u32, EngineError> {
        if *c + 4 > bytes.len() {
            return Err(EngineError::Corrupt("truncated length prefix".into()));
        }
        let v = u32::from_le_bytes(bytes[*c..*c + 4].try_into().unwrap());
        *c += 4;
        Ok(v)
    };

    let vec_len = read_u32(bytes, &mut c)? as usize;
    if c + vec_len * 4 > bytes.len() {
        return Err(EngineError::Corrupt("truncated vector".into()));
    }
    let mut vector = Vec::with_capacity(vec_len);
    for _ in 0..vec_len {
        vector.push(f32::from_le_bytes(bytes[c..c + 4].try_into().unwrap()));
        c += 4;
    }

    let text_len = read_u32(bytes, &mut c)? as usize;
    if c + text_len > bytes.len() {
        return Err(EngineError::Corrupt("truncated text".into()));
    }
    let text = String::from_utf8(bytes[c..c + text_len].to_vec())
        .map_err(|_| EngineError::Corrupt("text field is not valid utf-8".into()))?;
    c += text_len;

    let blob_len = read_u32(bytes, &mut c)? as usize;
    if c + blob_len > bytes.len() {
        return Err(EngineError::Corrupt("truncated blob".into()));
    }
    let blob = bytes[c..c + blob_len].to_vec();

    Ok((vector, text, blob))
}

pub struct Engine {
    docs: LsmEngine,
    vectors: VectorIndex,
    text: InvertedIndex,
}

impl Engine {
    /// Opens (or creates) the engine at `dir`. The vector index and the
    /// inverted index are pure in-memory derived state — only the LSM
    /// store is durable — so on open we replay the LSM's current snapshot
    /// (`LsmEngine::iter_all`) and rebuild both indexes from the stored
    /// document records. This is the same "durable log + rebuildable
    /// derived index" shape as the LSM's own WAL replay, one layer up.
    pub fn open(dir: impl AsRef<Path>, dim: usize) -> Result<Self, EngineError> {
        let docs = LsmEngine::open(dir)?;
        let mut vectors = VectorIndex::new(dim);
        let mut text = InvertedIndex::new();

        for (key, record) in docs.iter_all()? {
            if key.len() != 8 {
                continue; // not one of our doc-id keys; ignore
            }
            let id = u64::from_be_bytes(key.as_slice().try_into().unwrap());
            let (vector, doc_text, _blob) = decode_record(&record)?;
            vectors.insert(id, &vector)?;
            text.index_document(id, &doc_text);
        }

        Ok(Self {
            docs,
            vectors,
            text,
        })
    }

    /// Upsert: writing an existing `id` again replaces its vector, its
    /// text, and its blob. The LSM write happens first so that if it fails,
    /// the in-memory indexes are never updated (no partial state). On a
    /// non-crash IO failure the engine remains consistent.
    pub fn upsert(
        &mut self,
        id: u64,
        vector: &[f32],
        text: &str,
        blob: &[u8],
    ) -> Result<(), EngineError> {
        // Durable write first — in-memory indexes only updated on success so
        // a transient IO error doesn't leave state inconsistent.
        let record = encode_record(vector, text, blob);
        self.docs.put(&id.to_be_bytes(), &record)?;
        self.vectors.insert(id, vector)?;
        self.text.index_document(id, text);
        Ok(())
    }

    /// LSM tombstone written before clearing in-memory indexes. If the LSM
    /// write fails, memory stays consistent and the doc isn't silently lost.
    pub fn delete(&mut self, id: u64) -> Result<(), EngineError> {
        self.docs.delete(&id.to_be_bytes())?;
        self.vectors.remove(id);
        self.text.remove_document(id);
        Ok(())
    }

    pub fn get_blob(&self, id: u64) -> Result<Option<Vec<u8>>, EngineError> {
        match self.docs.get(&id.to_be_bytes())? {
            Some(record) => Ok(Some(decode_record(&record)?.2)),
            None => Ok(None),
        }
    }

    /// Hybrid query: vector similarity + BM25, fused by reciprocal rank.
    pub fn query(
        &self,
        vector: &[f32],
        text_query: &str,
        k: usize,
    ) -> Result<Vec<SearchHit>, EngineError> {
        let pool = (k * CANDIDATE_POOL_MULTIPLIER).max(MIN_CANDIDATE_POOL);
        let vector_results = self.vectors.search(vector, pool, Metric::Cosine)?;
        let text_results = self.text.search(text_query, pool);

        let fused = fusion::reciprocal_rank_fusion(&[&vector_results, &text_results], 60.0);

        let mut hits = Vec::with_capacity(k.min(fused.len()));
        for (id, score) in fused.into_iter().take(k) {
            let blob = match self.docs.get(&id.to_be_bytes())? {
                Some(record) => decode_record(&record)?.2,
                None => Vec::new(),
            };
            hits.push(SearchHit { id, score, blob });
        }
        Ok(hits)
    }

    pub fn flush(&mut self) -> Result<(), EngineError> {
        Ok(self.docs.flush()?)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let mut p = env::temp_dir();
        p.push(format!(
            "sift_core_engine_test_{}_{}",
            std::process::id(),
            name
        ));
        fs::remove_dir_all(&p).ok();
        p
    }

    #[test]
    fn hybrid_query_returns_and_fetches_blobs() {
        let dir = tmp_dir("hybrid");
        let mut engine = Engine::open(&dir, 3).unwrap();
        engine
            .upsert(1, &[1.0, 0.0, 0.0], "rust database engine", b"doc-one")
            .unwrap();
        engine
            .upsert(2, &[0.0, 1.0, 0.0], "python web framework", b"doc-two")
            .unwrap();
        engine
            .upsert(
                3,
                &[0.9, 0.1, 0.0],
                "rust systems programming",
                b"doc-three",
            )
            .unwrap();

        let hits = engine.query(&[1.0, 0.0, 0.0], "rust database", 2).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[0].blob, b"doc-one".to_vec());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delete_removes_from_both_indexes() {
        let dir = tmp_dir("delete");
        let mut engine = Engine::open(&dir, 2).unwrap();
        engine.upsert(1, &[1.0, 0.0], "hello world", b"x").unwrap();
        engine.delete(1).unwrap();
        let hits = engine.query(&[1.0, 0.0], "hello", 5).unwrap();
        assert!(hits.is_empty());
        assert_eq!(engine.get_blob(1).unwrap(), None);
        fs::remove_dir_all(&dir).ok();
    }
}
