# torg-core

Core runtime for **TØR-G** (Token-Only Reasoner - Graph) — a zero-parser, boolean-circuit IR for AI agent policy synthesis.

[![Crates.io](https://img.shields.io/crates/v/torg-core.svg)](https://crates.io/crates/torg-core)
[![Documentation](https://docs.rs/torg-core/badge.svg)](https://docs.rs/torg-core)

## Overview

TØR-G enables LLMs to generate **formally verifiable boolean circuits** by emitting a constrained token vocabulary. The deterministic state machine guarantees syntactically correct output through logit masking.

**Key Properties:**
- **No text parsing** — token stream only
- **Deterministic construction** — state machine compilation
- **Pure boolean combinatorics** — no arithmetic, state, or loops
- **DAG structure** — formally verifiable
- **Sub-microsecond evaluation** — ~139 ns for typical policies

## Usage

```rust
use torg_core::{Builder, Token, evaluate};
use std::collections::HashMap;

// Build: "Allow if Admin OR (Owner XOR Public)"
let mut builder = Builder::new();

// Declare inputs
builder.push(Token::InputDecl)?;
builder.push(Token::Id(0))?;  // is_admin
builder.push(Token::InputDecl)?;
builder.push(Token::Id(1))?;  // is_owner
builder.push(Token::InputDecl)?;
builder.push(Token::Id(2))?;  // is_public

// Node 3: owner XOR public
builder.push(Token::NodeStart)?;
builder.push(Token::Id(3))?;
builder.push(Token::Xor)?;
builder.push(Token::Id(1))?;
builder.push(Token::Id(2))?;
builder.push(Token::NodeEnd)?;

// Node 4: admin OR node(3)
builder.push(Token::NodeStart)?;
builder.push(Token::Id(4))?;
builder.push(Token::Or)?;
builder.push(Token::Id(0))?;
builder.push(Token::Id(3))?;
builder.push(Token::NodeEnd)?;

// Declare output
builder.push(Token::OutputDecl)?;
builder.push(Token::Id(4))?;

let graph = builder.finish()?;

// Evaluate
let inputs: HashMap<u16, bool> = [
    (0, false),  // not admin
    (1, true),   // is owner
    (2, false),  // not public
].into();

let outputs = evaluate(&graph, &inputs)?;
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

```
NOT(a)       = NOR(a, a)
AND(a, b)    = NOR(NOR(a,a), NOR(b,b))
IMPLIES(a,b) = OR(NOR(a,a), b)
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
