# Profiling this repo on Linux

Commands, not theory — copy/paste these against a `--release` build.

## 0. Build with symbols and native codegen

```bash
RUSTFLAGS="-C target-cpu=native -C debuginfo=2" cargo build --release
```

`target-cpu=native` matters specifically for `src/vector.rs`'s distance
kernels — without it LLVM targets a conservative baseline ISA and won't
emit AVX2/AVX-512 for the 8-wide accumulator loops.

## 1. Coarse timing: the bench binary

```bash
cargo run --release --bin sift-bench
```

Gives ms/query and writes/sec. Good for "did this change help or hurt"
before reaching for anything heavier.

## 2. CPU profiling: `perf record` + flamegraph

```bash
perf record -F 999 -g -- ./target/release/sift-bench
perf script | inferno-collapse-perf | inferno-flamegraph > flame.svg
```

(`inferno` is the Rust flamegraph toolchain; `cargo install inferno` or use
the classic `FlameGraph` Perl scripts if you already have them.) Look for:

* `VectorIndex::search` dominated by the distance kernel (`dot`/`l2_sq`)
  vs. dominated by the heap push/pop in top-k selection — tells you
  whether to optimize the kernel or lower `k`/switch to an approximate
  index.
* `SSTable::get` time split between `File::open`/seeking vs. the bloom
  filter check — if seeks dominate, that's the signal to move to mmap or
  add a block cache.

## 3. Syscall-level: `strace`

```bash
strace -c ./target/release/sift-bench 2>&1 | tail -20
```

Aggregated syscall counts/time. For this repo specifically, worth checking:
`fsync`/`fdatasync` count during `bench_lsm_writes` should equal the write
count (we `sync()` the WAL on every `put`) — if you change the durability
policy to batch syncs, this is how you'd confirm the syscall count actually
dropped.

## 4. eBPF: `bpftrace` for latency histograms

```bash
sudo bpftrace -e '
uprobe:./target/release/sift-bench:sift_core::vector::VectorIndex::search {
    @start[tid] = nsecs;
}
uretprobe:./target/release/sift-bench:sift_core::vector::VectorIndex::search /@start[tid]/ {
    @latency_ns = hist(nsecs - @start[tid]);
    delete(@start[tid]);
}'
```

Needs the release binary built with `debuginfo=2` (step 0) so the symbol
resolves. This is the tool for "what's the p99, not just the mean" —
`perf record` sampling can miss short-but-frequent functions; uprobes catch
every call.

## 5. Debugging: `gdb` / `rust-gdb`

```bash
rust-gdb --args ./target/debug/sift-core
(gdb) break sift_core::sstable::SSTable::get
(gdb) run
```

Use a `--debug` (unoptimized) build for this — `--release` inlines
aggressively enough that stepping through `sstable.rs`'s block-scan loop
gets confusing fast.

## 6. Memory layout sanity checks

```bash
cargo build --release && \
  size target/release/sift-core && \
  valgrind --tool=massif ./target/release/sift-bench
```

`massif` is the tool to confirm the SoA claim in `vector.rs`'s module docs
actually holds — heap snapshots should show one large contiguous
allocation growing for the `Vec<f32>` data buffer, not N small ones.
