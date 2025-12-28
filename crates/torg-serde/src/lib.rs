//! Serialization/deserialization for TØR-G graphs.
//!
//! This crate provides encoding and decoding of TØR-G graphs
//! to various formats:
//!
//! - **Binary**: Compact format (~7 bytes per node) for storage and transmission
//! - **JSON**: Human-readable format for debugging and inspection
//!
//! # Binary Format
//!
//! The binary format is designed for compactness and fast parsing:
//!
//! ```text
//! Header:     [0x54, 0x47] "TG" magic + version:u8 + flags:u8
//! Inputs:     count:u16 + [id:u16; count]
//! Nodes:      count:u16 + [Node; count]
//! Outputs:    count:u16 + [id:u16; count]
//! ```
//!
//! # Examples
//!
//! ```
//! use torg_core::{Graph, Node, BoolOp, Source};
//! use torg_serde::{to_bytes, from_bytes, to_json, from_json};
//!
//! // Create a simple graph
//! let graph = Graph {
//!     inputs: vec![0, 1],
//!     nodes: vec![Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1))],
//!     outputs: vec![2],
//! };
//!
//! // Binary serialization
//! let bytes = to_bytes(&graph);
//! let restored = from_bytes(&bytes).unwrap();
//! assert_eq!(graph, restored);
//!
//! // JSON serialization
//! let json = to_json(&graph);
//! let restored = from_json(&json).unwrap();
//! assert_eq!(graph, restored);
//! ```

mod binary;
mod error;
mod json;

pub use binary::{from_bytes, to_bytes};
pub use error::SerdeError;
pub use json::{from_json, to_json, to_json_compact};

// Re-export torg_core for convenience
pub use torg_core;
