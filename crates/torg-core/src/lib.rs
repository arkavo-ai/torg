//! TØR-G: Token-Only Reasoner (Graph)
//!
//! A zero-parser, boolean-circuit IR for AI agent policy synthesis.
//!
//! # Overview
//!
//! TØR-G provides a deterministic state machine for constructing and
//! evaluating boolean circuit DAGs from token streams. Key properties:
//!
//! - **No text parsing** — token stream only
//! - **Deterministic state machine** — guaranteed valid construction
//! - **Pure boolean combinatorics** — no arithmetic, state, or loops
//! - **DAG structure** — formally verifiable
//!
//! # Example
//!
//! ```
//! use torg_core::{Builder, Token, evaluate};
//! use std::collections::HashMap;
//!
//! // Build a simple OR circuit: output = input0 OR input1
//! let mut builder = Builder::new();
//!
//! // Declare inputs
//! builder.push(Token::InputDecl).unwrap();
//! builder.push(Token::Id(0)).unwrap();
//! builder.push(Token::InputDecl).unwrap();
//! builder.push(Token::Id(1)).unwrap();
//!
//! // Define node: node(2) = input(0) OR input(1)
//! builder.push(Token::NodeStart).unwrap();
//! builder.push(Token::Id(2)).unwrap();
//! builder.push(Token::Or).unwrap();
//! builder.push(Token::Id(0)).unwrap();
//! builder.push(Token::Id(1)).unwrap();
//! builder.push(Token::NodeEnd).unwrap();
//!
//! // Declare output
//! builder.push(Token::OutputDecl).unwrap();
//! builder.push(Token::Id(2)).unwrap();
//!
//! let graph = builder.finish().unwrap();
//!
//! // Evaluate with inputs
//! let inputs: HashMap<u16, bool> = [(0, false), (1, true)].into();
//! let outputs = evaluate(&graph, &inputs).unwrap();
//! assert_eq!(outputs[&2], true); // false OR true = true
//! ```
//!
//! # LLM Integration
//!
//! The [`Builder::valid_next_tokens`] method returns the set of tokens
//! valid in the current state. This enables logit masking during LLM
//! inference, guaranteeing syntactically correct output:
//!
//! ```
//! use torg_core::{Builder, Token};
//!
//! let mut builder = Builder::new();
//! builder.push(Token::InputDecl).unwrap();
//! builder.push(Token::Id(0)).unwrap();
//! builder.push(Token::NodeStart).unwrap();
//! builder.push(Token::Id(1)).unwrap();
//!
//! // At this point, only operators are valid
//! let valid = builder.valid_next_tokens();
//! assert!(valid.contains(&Token::Or));
//! assert!(valid.contains(&Token::Nor));
//! assert!(valid.contains(&Token::Xor));
//! assert!(!valid.contains(&Token::NodeEnd));
//! ```

pub mod builder;
pub mod error;
pub mod eval;
pub mod graph;
pub mod limits;
pub mod token;

pub use builder::{Builder, Phase};
pub use error::{BuildError, EvalError};
pub use eval::evaluate;
pub use graph::{Graph, Node};
pub use limits::Limits;
pub use token::{BoolOp, Source, Token};
