//! LLM logit masking for TØR-G constrained decoding.
//!
//! This crate provides utilities for integrating TØR-G's
//! `valid_next_tokens()` with LLM inference engines.
//!
//! # Overview
//!
//! TØR-G graphs are constructed by feeding tokens to a `Builder`.
//! At each step, only certain tokens are valid (returned by
//! `Builder::valid_next_tokens()`). This crate provides:
//!
//! - **Token mapping**: Bidirectional mapping between TØR-G tokens and LLM vocab IDs
//! - **Logit masking**: Efficiently mask invalid tokens in logit vectors
//! - **Constrained decoding**: High-level orchestrator for decode loops
//!
//! # Example
//!
//! ```
//! use torg_mask::{ConstrainedDecoder, MaskGenerator, TokenMapping};
//!
//! // Create a token mapping (in production, map to actual LLM vocab IDs)
//! let mapping = TokenMapping::sequential(256);
//! let generator = MaskGenerator::new(mapping.clone(), 1000);
//! let mut decoder = ConstrainedDecoder::new(generator);
//!
//! // Simulate a decode loop
//! let token_ids = [
//!     5,   // InputDecl
//!     9,   // Id(0)
//!     3,   // NodeStart
//!     10,  // Id(1)
//!     0,   // Or
//!     9,   // Id(0)
//!     7,   // True
//!     4,   // NodeEnd
//!     6,   // OutputDecl
//!     10,  // Id(1)
//! ];
//!
//! for &id in &token_ids {
//!     let mask = decoder.next_mask();
//!     assert!(mask.is_allowed(id));
//!     decoder.feed_token(id).unwrap();
//! }
//!
//! let graph = decoder.finish().unwrap();
//! assert_eq!(graph.nodes.len(), 1);
//! ```

pub mod decoder;
pub mod generator;
pub mod mapping;
pub mod mask;

pub use decoder::{ConstrainedDecoder, DecodeError};
pub use generator::MaskGenerator;
pub use mapping::{TokenMapping, TokenMappingBuilder};
pub use mask::LogitMask;

// Re-export torg_core for convenience
pub use torg_core;
