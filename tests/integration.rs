use sift_core::Engine;
use std::fs;

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sift_core_integration_{}_{}",
        std::process::id(),
        name
    ));
    fs::remove_dir_all(&p).ok();
    p
}

#[test]
fn hybrid_search_survives_a_restart() {
    let dir = tmp_dir("restart");

    {
        let mut engine = Engine::open(&dir, 3).unwrap();
        engine
            .upsert(
                1,
                &[1.0, 0.0, 0.0],
                "hybrid search engine internals",
                b"blob-1",
            )
            .unwrap();
        engine
            .upsert(2, &[0.0, 1.0, 0.0], "unrelated cooking recipes", b"blob-2")
            .unwrap();
        engine
            .upsert(
                3,
                &[0.95, 0.05, 0.0],
                "search engine performance tuning",
                b"blob-3",
            )
            .unwrap();
        // engine dropped without an explicit flush() on the doc store; the
        // WAL is the only thing keeping blob-1/2/3 durable at this point.
    }

    let engine = Engine::open(&dir, 3).unwrap();
    let hits = engine.query(&[1.0, 0.0, 0.0], "search engine", 2).unwrap();

    assert_eq!(hits.len(), 2);
    let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    assert!(ids.contains(&1));
    assert!(ids.contains(&3));
    assert!(!ids.contains(&2));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn deleting_a_document_removes_it_from_hybrid_results_but_not_others() {
    let dir = tmp_dir("delete_hybrid");
    let mut engine = Engine::open(&dir, 2).unwrap();

    engine
        .upsert(1, &[1.0, 0.0], "alpha document about databases", b"a")
        .unwrap();
    engine
        .upsert(2, &[1.0, 0.0], "beta document about databases", b"b")
        .unwrap();
    engine.delete(1).unwrap();

    let hits = engine.query(&[1.0, 0.0], "databases", 5).unwrap();
    let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    assert_eq!(ids, vec![2]);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn upsert_overwrites_prior_vector_text_and_blob() {
    let dir = tmp_dir("upsert_overwrite");
    let mut engine = Engine::open(&dir, 2).unwrap();

    // A stable distractor doc lets us tell "doc 1 ranked because it
    // genuinely matches" apart from "doc 1 showed up because it's the only
    // vector in the index" (with one doc, top-k always returns it).
    engine
        .upsert(
            2,
            &[1.0, 0.0],
            "a distractor document about cars and trucks",
            b"distractor",
        )
        .unwrap();

    engine
        .upsert(1, &[1.0, 0.0], "old text about cars", b"old-blob")
        .unwrap();
    engine
        .upsert(1, &[0.0, 1.0], "new text about boats", b"new-blob")
        .unwrap();

    assert_eq!(engine.get_blob(1).unwrap(), Some(b"new-blob".to_vec()));

    // Querying for doc 1's *old* vector direction + old text term should
    // now favor the distractor, which still genuinely matches both.
    let hits = engine.query(&[1.0, 0.0], "cars", 5).unwrap();
    assert_eq!(
        hits[0].id,
        2,
        "distractor should now outrank doc 1 on its old attributes: {:?}",
        hits.iter().map(|h| h.id).collect::<Vec<_>>()
    );

    // The new attributes do match doc 1, and doc 1 alone.
    let hits = engine.query(&[0.0, 1.0], "boats", 5).unwrap();
    assert_eq!(hits[0].id, 1);

    fs::remove_dir_all(&dir).ok();
}
