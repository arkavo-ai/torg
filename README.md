# TØR-G

**Token-Only Reasoner (Graph)** — A zero-parser, boolean-circuit IR for AI agent policy synthesis.

[![CI](https://github.com/arkavo-ai/torg/actions/workflows/ci.yml/badge.svg)](https://github.com/arkavo-ai/torg/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/torg-core.svg)](https://crates.io/crates/torg-core)
[![Documentation](https://docs.rs/torg-core/badge.svg)](https://docs.rs/torg-core)

## Overview

TØR-G enables LLMs to generate **formally verifiable boolean circuits** by emitting a constrained token vocabulary. The deterministic state machine guarantees syntactically correct output through logit masking.

**Key Properties:**
- **No text parsing** — token stream only
- **Deterministic construction** — state machine compilation
- **Pure boolean combinatorics** — no arithmetic, state, or loops
- **DAG structure** — formally verifiable

## Crates

| Crate | Description |
|-------|-------------|
| [`torg-core`](crates/torg-core) | Core runtime: tokens, builder, evaluator |
| [`torg-serde`](crates/torg-serde) | Serialization/deserialization (planned) |
| [`torg-verify`](crates/torg-verify) | SAT/BDD verification (planned) |
| [`torg-mask`](crates/torg-mask) | LLM logit masking: constrained decoding |

## Quick Start

```rust
use torg_core::{Builder, Token, evaluate};
use std::collections::HashMap;

// Build: "Allow if Admin OR (Owner XOR Public)"
let mut builder = Builder::new();

// Declare inputs
builder.push(Token::InputDecl).unwrap();
builder.push(Token::Id(0)).unwrap();  // is_admin
builder.push(Token::InputDecl).unwrap();
builder.push(Token::Id(1)).unwrap();  // is_owner
builder.push(Token::InputDecl).unwrap();
builder.push(Token::Id(2)).unwrap();  // is_public

// Node 3: owner XOR public
builder.push(Token::NodeStart).unwrap();
builder.push(Token::Id(3)).unwrap();
builder.push(Token::Xor).unwrap();
builder.push(Token::Id(1)).unwrap();
builder.push(Token::Id(2)).unwrap();
builder.push(Token::NodeEnd).unwrap();

// Node 4: admin OR node(3)
builder.push(Token::NodeStart).unwrap();
builder.push(Token::Id(4)).unwrap();
builder.push(Token::Or).unwrap();
builder.push(Token::Id(0)).unwrap();
builder.push(Token::Id(3)).unwrap();
builder.push(Token::NodeEnd).unwrap();

// Declare output
builder.push(Token::OutputDecl).unwrap();
builder.push(Token::Id(4)).unwrap();

let graph = builder.finish().unwrap();

// Evaluate
let inputs: HashMap<u16, bool> = [
    (0, false),  // not admin
    (1, true),   // is owner
    (2, false),  // not public
].into();

let outputs = evaluate(&graph, &inputs).unwrap();
assert!(outputs[&4]);  // owner XOR !public = true
```

## LLM Integration

The `valid_next_tokens()` method enables constrained decoding:

```rust
use torg_core::{Builder, Token};

let mut builder = Builder::new();
builder.push(Token::InputDecl).unwrap();
builder.push(Token::Id(0)).unwrap();
builder.push(Token::NodeStart).unwrap();
builder.push(Token::Id(1)).unwrap();

// Only operators are valid here
let valid = builder.valid_next_tokens();
assert!(valid.contains(&Token::Or));
assert!(valid.contains(&Token::Nor));
assert!(valid.contains(&Token::Xor));
assert!(!valid.contains(&Token::NodeEnd));  // Not yet!
```

Use these tokens to mask LLM logits during inference, guaranteeing valid circuit generation.

## Token Vocabulary

| Token | Symbol | Purpose |
|-------|--------|---------|
| `Or` | ∨ | Logical OR |
| `Nor` | ⊽ | Logical NOR |
| `Xor` | ⊻ | Logical XOR |
| `NodeStart` | ● | Begin node definition |
| `NodeEnd` | ○ | End node definition |
| `InputDecl` | ◎IN | Declare input |
| `OutputDecl` | ◎OUT | Declare output |
| `True` | ◎T | Constant true |
| `False` | ◎F | Constant false |
| `Id(n)` | — | Identifier (0..65535) |

## Derived Operations

AND, NOT, NAND, IMPLIES can be derived from the base operators:

```
NOT(a)       = NOR(a, a)
AND(a, b)    = NOR(NOR(a,a), NOR(b,b))
IMPLIES(a,b) = OR(NOR(a,a), b)
```

## Specification

See [`spec/draft-arkavo-torg-decision-00.md`](spec/draft-arkavo-torg-decision-00.md) for the full specification.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
