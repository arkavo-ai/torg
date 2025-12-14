//! Benchmarks for TØR-G evaluation performance.
//!
//! Target: sub-microsecond evaluation for typical policy graphs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use torg_core::{evaluate, Builder, Graph, Limits, Token};

/// Build a chain of OR nodes for benchmarking.
fn build_chain_graph(depth: usize) -> Graph {
    use Token::*;

    let mut builder = Builder::with_limits(Limits::permissive());

    // Single input
    builder.push(InputDecl).unwrap();
    builder.push(Id(0)).unwrap();

    // Chain of OR nodes
    for i in 1..=depth as u16 {
        builder.push(NodeStart).unwrap();
        builder.push(Id(i)).unwrap();
        builder.push(Or).unwrap();
        if i == 1 {
            builder.push(Id(0)).unwrap(); // First node references input
        } else {
            builder.push(Id(i - 1)).unwrap(); // Reference previous node
        }
        builder.push(True).unwrap();
        builder.push(NodeEnd).unwrap();
    }

    builder.push(OutputDecl).unwrap();
    builder.push(Id(depth as u16)).unwrap();

    builder.finish().unwrap()
}

/// Build a wide graph (many parallel nodes) for benchmarking.
fn build_wide_graph(width: usize) -> Graph {
    use Token::*;

    let mut builder = Builder::with_limits(Limits::permissive());

    // Multiple inputs
    for i in 0..width as u16 {
        builder.push(InputDecl).unwrap();
        builder.push(Id(i)).unwrap();
    }

    // One node per input pair (width/2 nodes)
    let base_node_id = width as u16;
    for i in 0..(width / 2) as u16 {
        builder.push(NodeStart).unwrap();
        builder.push(Id(base_node_id + i)).unwrap();
        builder.push(Or).unwrap();
        builder.push(Id(i * 2)).unwrap();
        builder.push(Id(i * 2 + 1)).unwrap();
        builder.push(NodeEnd).unwrap();
    }

    // Output all nodes
    for i in 0..(width / 2) as u16 {
        builder.push(OutputDecl).unwrap();
        builder.push(Id(base_node_id + i)).unwrap();
    }

    builder.finish().unwrap()
}

/// Build the "Admin OR (Owner XOR Public)" policy graph.
fn build_policy_graph() -> Graph {
    use Token::*;

    let mut builder = Builder::new();

    builder.push(InputDecl).unwrap();
    builder.push(Id(0)).unwrap(); // admin
    builder.push(InputDecl).unwrap();
    builder.push(Id(1)).unwrap(); // owner
    builder.push(InputDecl).unwrap();
    builder.push(Id(2)).unwrap(); // public

    // XOR node
    builder.push(NodeStart).unwrap();
    builder.push(Id(3)).unwrap();
    builder.push(Xor).unwrap();
    builder.push(Id(1)).unwrap();
    builder.push(Id(2)).unwrap();
    builder.push(NodeEnd).unwrap();

    // OR node
    builder.push(NodeStart).unwrap();
    builder.push(Id(4)).unwrap();
    builder.push(Or).unwrap();
    builder.push(Id(0)).unwrap();
    builder.push(Id(3)).unwrap();
    builder.push(NodeEnd).unwrap();

    builder.push(OutputDecl).unwrap();
    builder.push(Id(4)).unwrap();

    builder.finish().unwrap()
}

fn bench_eval_chain(c: &mut Criterion) {
    let graph_10 = build_chain_graph(10);
    let graph_50 = build_chain_graph(50);
    let graph_100 = build_chain_graph(100);
    let inputs: HashMap<u16, bool> = [(0, true)].into();

    c.bench_function("eval_chain_depth_10", |b| {
        b.iter(|| evaluate(black_box(&graph_10), black_box(&inputs)))
    });

    c.bench_function("eval_chain_depth_50", |b| {
        b.iter(|| evaluate(black_box(&graph_50), black_box(&inputs)))
    });

    c.bench_function("eval_chain_depth_100", |b| {
        b.iter(|| evaluate(black_box(&graph_100), black_box(&inputs)))
    });
}

fn bench_eval_wide(c: &mut Criterion) {
    let graph = build_wide_graph(100);
    let inputs: HashMap<u16, bool> = (0..100u16).map(|i| (i, i % 2 == 0)).collect();

    c.bench_function("eval_wide_50_nodes", |b| {
        b.iter(|| evaluate(black_box(&graph), black_box(&inputs)))
    });
}

fn bench_eval_policy(c: &mut Criterion) {
    let graph = build_policy_graph();
    let inputs: HashMap<u16, bool> = [(0, true), (1, false), (2, true)].into();

    c.bench_function("eval_policy_admin_or_xor", |b| {
        b.iter(|| evaluate(black_box(&graph), black_box(&inputs)))
    });
}

fn bench_build(c: &mut Criterion) {
    use Token::*;

    // Build tokens for a 50-node chain
    let tokens: Vec<Token> = {
        let mut t = vec![InputDecl, Id(0)];
        for i in 1..=50u16 {
            t.extend([NodeStart, Id(i), Or, Id(i - 1), True, NodeEnd]);
        }
        t.extend([OutputDecl, Id(50)]);
        t
    };

    c.bench_function("build_50_nodes", |b| {
        b.iter(|| {
            let mut builder = Builder::with_limits(Limits::permissive());
            for &token in black_box(&tokens) {
                builder.push(token).unwrap();
            }
            builder.finish().unwrap()
        })
    });
}

fn bench_valid_next_tokens(c: &mut Criterion) {
    // Builder in various states
    let mut builder_inputs = Builder::new();
    for i in 0..10u16 {
        builder_inputs.push(Token::InputDecl).unwrap();
        builder_inputs.push(Token::Id(i)).unwrap();
    }

    c.bench_function("valid_next_tokens_10_inputs", |b| {
        b.iter(|| black_box(&builder_inputs).valid_next_tokens())
    });
}

criterion_group!(
    benches,
    bench_eval_chain,
    bench_eval_wide,
    bench_eval_policy,
    bench_build,
    bench_valid_next_tokens,
);

criterion_main!(benches);
