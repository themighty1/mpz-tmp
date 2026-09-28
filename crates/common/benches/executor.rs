//! Executor benchmarks.
//!
//! Measures the `mpz-common` task executor under load: sub-task scheduling
//! (`join`), work distribution with real CPU work (`map`), and mux-channel I/O
//! against a cooperative peer.
//!
//! # Model
//!
//! There is exactly **one** loaded executor, sized to all cores. The peer
//! endpoint in the I/O benchmark is a *cooperative* context with no worker
//! threads, so the only thread pool in the process is the executor under test.
//! All I/O runs over an in-memory mock [`Mux`].
//!
//! Run with:
//!
//! ```text
//! cargo bench -p mpz-common --bench executor
//! ```
//!
//! # Executor under test
//!
//! This targets the current executor: a [`Session`] backed by a [`ThreadPool`]
//! sized to all cores. Only [`Runtime`], [`build_loaded`] and [`build_peer`]
//! below are version-specific; everything else uses the shared [`Context`]
//! API. To target the legacy mux-coupled executor, replace them with:
//!
//! ```text
//! type Runtime = mpz_common::Executor;
//!
//! fn build_loaded(mux: impl Mux + Send + Sync + 'static) -> Runtime {
//!     mpz_common::Executor::builder().num_threads(cores()).build(mux)
//! }
//!
//! fn build_peer(mux: impl Mux + Send + Sync + 'static) -> Context {
//!     Context::new(mux).expect("context should build")
//! }
//! ```

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

/// Number of worker threads for the loaded executor.
fn cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

/// Deterministic CPU work: `rounds` iterations of a loop-carried mix.
///
/// The loop-carried dependency plus `black_box` keeps the optimizer from
/// eliding or vectorizing it, so it represents real per-item compute.
fn burn(seed: u64, rounds: u32) -> u64 {
    let mut v = seed;
    for _ in 0..rounds {
        v = v.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(27) ^ v;
    }
    black_box(v)
}

// ============================================================================
// Executor under test (the only version-specific code).
// ============================================================================

/// The loaded executor under test.
type Runtime = mpz_common::Session;

/// Builds the executor under test, sized to all cores.
fn build_loaded(mux: impl Mux + Send + Sync + 'static) -> Runtime {
    let pool = mpz_common::ThreadPool::builder()
        .num_threads(cores())
        .build()
        .expect("thread pool should build");

    mpz_common::Session::builder()
        .pool(pool)
        .build(mux)
        .expect("session should build")
}

/// Builds the peer endpoint: a cooperative context with no worker threads.
fn build_peer(mux: impl Mux + Send + Sync + 'static) -> Context {
    mpz_common::Session::builder()
        .cooperative()
        .build(mux)
        .expect("peer session should build")
        .new_context()
        .expect("peer context should build")
}

// ============================================================================
// Mock mux.
// ============================================================================

/// An in-memory [`Mux`] shared by two endpoints.
///
/// The first `open` of an id creates a `tokio` duplex and parks its peer; the
/// second `open` of that id claims the parked peer. Both endpoints must open
/// the same ids in the same way (which the executor guarantees for a shared
/// context fork sequence).
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
/// Dropping the peer immediately keeps the scheduling benchmarks free of parked
/// channels.
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

    let runtime = build_loaded(LocalMux);
    let mut ctx = runtime.new_context().unwrap();

    group.bench_function("join", |b| {
        b.iter(|| {
            futures::executor::block_on(ctx.join(
                |_ctx| async move { black_box(1u64) }.boxed(),
                |_ctx| async move { black_box(2u64) }.boxed(),
            ))
            .unwrap()
        });
    });

    group.finish();
}

/// Work distribution: apply a function of tunable CPU cost to `n` items.
///
/// `rounds = 0` is the dispatch floor; larger `rounds` loads the executor with
/// real per-item compute.
fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor/map");

    for &items in &[4096usize, 8192] {
        group.throughput(Throughput::Elements(items as u64));

        let runtime = build_loaded(LocalMux);
        let mut ctx = runtime.new_context().unwrap();

        for &rounds in &[0u32, 1_000] {
            let id = BenchmarkId::new(format!("{items}i"), format!("{rounds}r"));
            group.bench_with_input(id, &(items, rounds), |b, &(items, rounds)| {
                b.iter(|| {
                    let items: Vec<u64> = (0..items as u64).collect();
                    futures::executor::block_on(ctx.map(items, move |_ctx, x| {
                        async move { burn(x, rounds) }.boxed()
                    }))
                    .unwrap()
                });
            });
        }
    }

    group.finish();
}

/// I/O: the loaded executor runs one endpoint while a cooperative peer replies.
///
/// Each item round-trips a message over a mux channel. The peer has no worker
/// threads, so the only executor in play is the one under test.
fn bench_map_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor/map_io");
    group.sample_size(20);

    for &items in &[16usize, 256] {
        group.throughput(Throughput::Elements(items as u64));

        let (mux_a, mux_b) = MockMux::pair();
        let runtime = build_loaded(mux_a);
        let mut ctx_a = runtime.new_context().unwrap();
        let mut ctx_b = build_peer(mux_b);

        group.bench_with_input(BenchmarkId::from_parameter(items), &items, |b, &items| {
            b.iter(|| {
                let items_a: Vec<u64> = (0..items as u64).collect();
                let items_b: Vec<u64> = (0..items as u64).collect();

                let (ra, rb) = futures::executor::block_on(async {
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
                ra.unwrap();
                rb.unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_join, bench_map, bench_map_io);
criterion_main!(benches);
