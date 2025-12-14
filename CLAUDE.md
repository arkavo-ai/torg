# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build --workspace        # Build all crates
cargo test --workspace         # Run all tests
cargo clippy --workspace       # Lint all crates
cargo fmt --all --check        # Check formatting
cargo bench -p torg-core       # Run benchmarks
```

Single test: `cargo test -p torg-core test_name`

## Architecture

TØR-G is a boolean-circuit IR for AI agent policy synthesis. LLMs emit a constrained token vocabulary; the deterministic state machine guarantees syntactically correct DAGs via logit masking.

### Workspace Structure

- **torg-core** — Complete implementation: tokens, builder state machine, evaluator
- **torg-serde** — Placeholder for serialization
- **torg-verify** — Placeholder for SAT/BDD verification
- **torg-mask** — Placeholder for LLM logit masking integration

### torg-core Modules

| Module | Purpose |
|--------|---------|
| `token.rs` | Token enum (Or, Nor, Xor, NodeStart, NodeEnd, etc.), Source, BoolOp |
| `builder.rs` | State machine: Phase (Inputs→Nodes→Outputs→Done) + NodeState for node parsing |
| `graph.rs` | Graph/Node structs, depth calculation |
| `eval.rs` | `evaluate()` function — processes graph with input values |
| `limits.rs` | DoS prevention: max_nodes, max_depth, max_inputs, max_outputs |
| `error.rs` | BuildError, EvalError using thiserror |

### Key Design Patterns

**Builder State Machine**: Two-level state — `Phase` for top-level (Inputs/Nodes/Outputs/Done), `NodeState` for within node definitions. `valid_next_tokens()` returns valid tokens for current state (critical for LLM logit masking).

**DAG Enforcement**: IDs tracked in `defined: HashSet<u16>`. References must be to already-defined IDs, preventing forward references and guaranteeing DAG structure.

**Binary Nodes**: Each node has exactly 2 operands (left, right) with explicit IDs in token stream.

## Token Grammar

```
graph   = inputs nodes outputs
inputs  = (InputDecl Id)*
nodes   = (NodeStart Id Op Source Source NodeEnd)*
outputs = (OutputDecl Id)+
```

Derived ops: `NOT(a) = NOR(a,a)`, `AND(a,b) = NOR(NOR(a,a), NOR(b,b))`
