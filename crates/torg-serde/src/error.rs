//! Error types for TØR-G serialization.

use thiserror::Error;

/// Errors that can occur during serialization/deserialization.
#[derive(Error, Debug)]
pub enum SerdeError {
    /// Invalid magic bytes in binary format.
    #[error("invalid magic bytes: expected 'TG', got {0:02X} {1:02X}")]
    InvalidMagic(u8, u8),

    /// Unsupported binary format version.
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u8),

    /// Invalid operation code in binary format.
    #[error("invalid op code: {0}")]
    InvalidOpCode(u8),

    /// Invalid source tag in binary format.
    #[error("invalid source tag: {0}")]
    InvalidSourceTag(u8),

    /// Unexpected end of input while reading binary format.
    #[error("unexpected end of input at offset {0}")]
    UnexpectedEof(usize),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Graph validation error after deserialization.
    #[error("invalid graph: {0}")]
    InvalidGraph(String),
}
