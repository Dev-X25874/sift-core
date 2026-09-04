use sift_core::Engine;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = env::temp_dir().join(format!("sift_core_demo_{}", std::process::id()));
    let mut engine = Engine::open(&dir, 4)?;

    let corpus: &[(u64, [f32; 4], &str, &str)] = &[
        (
            1,
            [1.0, 0.1, 0.0, 0.0],
            "sift-core is a hybrid search engine for vectors and full text",
            "doc 1",
        ),
        (
            2,
            [0.9, 0.2, 0.1, 0.0],
            "LSM trees power most modern storage engines",
            "doc 2",
        ),
        (
            3,
            [0.0, 0.0, 1.0, 1.0],
            "sourdough bread recipes and baking tips",
            "doc 3",
        ),
        (
            4,
            [0.8, 0.3, 0.0, 0.1],
            "rust is a systems programming language with no GC",
            "doc 4",
        ),
    ];

    for (id, vec, text, blob) in corpus {
        engine.upsert(*id, vec, text, blob.as_bytes())?;
    }
    engine.flush()?;

    println!("indexed {} documents\n", engine.len());

    let query_vec = [1.0, 0.1, 0.0, 0.0];
    let hits = engine.query(&query_vec, "search engine storage", 3)?;

    println!("query: vector~doc1, text=\"search engine storage\"");
    for hit in hits {
        println!(
            "  id={:<3} score={:>7.4}  blob={:?}",
            hit.id,
            hit.score,
            String::from_utf8_lossy(&hit.blob)
        );
    }

    Ok(())
}
