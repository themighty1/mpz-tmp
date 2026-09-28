//! Executor benchmarks.
//!
//! Measures the cost of the `mpz-common` task executor: sub-task scheduling
//! (`join`), work distribution (`map`) and mux-channel I/O between two parties.
//!
//! All I/O runs over an in-memory mock [`Mux`], so the benchmarks need no
//! sockets and can be driven from a single process. The mock is intentionally
//! defined in this file (rather than reusing `mux::test_framed_mux`) so that it
//! also compiles against the previous, mux-coupled executor.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p mpz-common --bench executor
//! ```
//!
//! # Backporting to the previous executor
//!
//! The current executor is split into [`Session`] + [`ThreadPool`]. The
//! previous iteration was a single mux-coupled `mpz_common::Executor`. Only
//! the [`Runtime`] alias and [`build_runtime`] helper below are
//! version-specific; everything else uses the shared [`Context`] API. To run
//! this benchmark against the old executor, replace those two items with:
//!
//! ```text
//! type Runtime = mpz_common::Executor;
//!
//! fn build_runtime(num_threads: usize, mux: impl Mux + Send + Sync + 'static) -> Runtime {
//!     mpz_common::Executor::builder().num_threads(num_threads).build(mux)
//! }
//! ```
//!
//! This works because the old `Executor` is generic over the same [`Mux`] trait
//! and `new_context` hands back the same [`Context`] type; the mock mux below
//! satisfies its `M: Mux` bound.

use std::{
    collections::{HashMap, HashSet},
    hint::black_box,
    io,
    sync::{Arc, Mutex},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::{FutureExt, future};
use mpz_common::{Context, mux::Mux};
use serio::{SinkExt, StreamExt};
use tokio::io::DuplexStream;
use tokio_util::compat::TokioAsyncReadCompatExt;

// ============================================================================
// Executor under test (the only version-specific code).
// ============================================================================

/// The executor type under test.
///
/// New executor: `mpz_common::Session` backed by a dedicated `ThreadPool`.
type Runtime = mpz_common::Session;

/// Builds an executor with `num_threads` worker threads and the given mux.
///
/// New executor: assemble a `ThreadPool` and hand it to a `Session`. The old,
/// mux-coupled executor built itself directly from the mux; see the module
/// docs for the backport.
fn build_runtime(num_threads: usize, mux: impl Mux + Send + Sync + 'static) -> Runtime {
    let pool = mpz_common::ThreadPool::builder()
        .num_threads(num_threads)
        .build()
        .expect("thread pool should build");

    mpz_common::Session::builder()
        .pool(pool)
        .build(mux)
        .expect("session should build")
}

// ============================================================================
// Mock mux.
// ============================================================================

/// An in-memory [`Mux`] shared by two parties.
///
/// The first `open` of an id creates a `tokio` duplex and parks its peer; the
/// second `open` of that id claims the parked peer. Both parties must therefore
/// open the same ids in the same way (which the executor guarantees for a
/// shared context fork sequence).
#[derive(Clone)]
struct MockMux {
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    opened: HashSet<Vec<u8>>,
    waiting: HashMap<Vec<u8>, DuplexStream>,
}

impl MockMux {
    /// Creates a connected pair of mock muxes.
    fn pair() -> (Self, Self) {
        let state = Arc::new(Mutex::new(State::default()));
        (
            Self {
                state: state.clone(),
            },
            Self { state },
        )
    }
}

impl Mux for MockMux {
    fn open(&self, id: &[u8]) -> Result<mpz_common::io::Io, io::Error> {
        let mut state = self.state.lock().unwrap();

        if let Some(peer) = state.waiting.remove(id) {
            Ok(mpz_common::io::Io::from_io(peer.compat()))
        } else if state.opened.insert(id.to_vec()) {
            let (local, peer) = tokio::io::duplex(1 << 12);
            state.waiting.insert(id.to_vec(), peer);
            Ok(mpz_common::io::Io::from_io(local.compat()))
        } else {
            Err(io::Error::other("duplicate stream id"))
        }
    }
}

/// A [`Mux`] whose channels are never used.
///
/// Dropping the peer immediately keeps the scheduling benchmarks allocation
/// free of parked channels.
struct LocalMux;

impl Mux for LocalMux {
    fn open(&self, _id: &[u8]) -> Result<mpz_common::io::Io, io::Error> {
        let (local, _peer) = tokio::io::duplex(1);
        Ok(mpz_common::io::Io::from_io(local.compat()))
    }
}

// ============================================================================
// Benchmarks.
// ============================================================================

/// Scheduling overhead: fork a context and join two trivial sub-tasks.
fn bench_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor/join");

    for &threads in &[1usize, 2, 4] {
        let runtime = build_runtime(threads, LocalMux);
        let mut ctx = runtime.new_context().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter(|| {
                futures::executor::block_on(ctx.join(
                    |_ctx| async move { black_box(1u64) }.boxed(),
                    |_ctx| async move { black_box(2u64) }.boxed(),
                ))
                .unwrap()
            });
        });
    }

    group.finish();
}

/// Work distribution: apply a trivial function to `n` items via `map`.
fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor/map");

    for &items in &[16usize, 256, 4096] {
        group.throughput(Throughput::Elements(items as u64));

        let runtime = build_runtime(4, LocalMux);
        let mut ctx = runtime.new_context().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(items), &items, |b, &items| {
            b.iter(|| {
                let items: Vec<u64> = (0..items as u64).collect();
                futures::executor::block_on(
                    ctx.map(items, |_ctx, x| async move { black_box(x + 1) }.boxed()),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

/// Two-party `map` where each item round-trips a message over a mux channel.
///
/// This is the closest to protocol execution: it exercises sub-task wakeups,
/// cross-thread scheduling and framed channel I/O.
fn bench_map_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor/map_io");
    group.sample_size(20);

    for &threads in &[1usize, 2, 4] {
        for &items in &[16usize, 256] {
            group.throughput(Throughput::Elements(items as u64));

            let (mux_a, mux_b) = MockMux::pair();
            let runtime_a = build_runtime(threads, mux_a);
            let runtime_b = build_runtime(threads, mux_b);

            let mut ctx_a = runtime_a.new_context().unwrap();
            let mut ctx_b = runtime_b.new_context().unwrap();

            let id = BenchmarkId::new(format!("{threads}t"), items);
            group.bench_with_input(id, &items, |b, &items| {
                b.iter(|| {
                    let items_a: Vec<u64> = (0..items as u64).collect();
                    let items_b: Vec<u64> = (0..items as u64).collect();

                    futures::executor::block_on(async {
                        future::join(
                            ctx_a.map(items_a, |ctx, x| {
                                async move {
                                    ctx.io_mut().send(x).await.unwrap();
                                }
                                .boxed()
                            }),
                            ctx_b.map(items_b, |ctx, x| {
                                async move {
                                    let received: u64 = ctx.io_mut().next().await.unwrap().unwrap();
                                    black_box(received + x)
                                }
                                .boxed()
                            }),
                        )
                        .await
                    });
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_join, bench_map, bench_map_io);
criterion_main!(benches);
