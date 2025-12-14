//! Error types for TØR-G graph construction and evaluation.

use thiserror::Error;

/// Errors that can occur during graph construction.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// Token not valid in current state.
    #[error("unexpected token in current state")]
    UnexpectedToken,

    /// Attempted to define an ID that already exists.
    #[error("duplicate definition of ID {0}")]
    DuplicateDefinition(u16),

    /// Referenced an ID that hasn't been defined yet.
    #[error("undefined reference to ID {0}")]
    UndefinedReference(u16),

    /// Node definition was not completed (missing NodeEnd).
    #[error("node definition incomplete")]
    IncompleteNode,

    /// Graph has no declared outputs.
    #[error("graph has no outputs")]
    NoOutputs,

    /// Maximum node count exceeded.
    #[error("maximum nodes exceeded ({0})")]
    MaxNodesExceeded(usize),

    /// Maximum graph depth exceeded.
    #[error("maximum depth exceeded ({0})")]
    MaxDepthExceeded(usize),

    /// Maximum input count exceeded.
    #[error("maximum inputs exceeded ({0})")]
    MaxInputsExceeded(usize),

    /// Maximum output count exceeded.
    #[error("maximum outputs exceeded ({0})")]
    MaxOutputsExceeded(usize),
}

/// Errors that can occur during graph evaluation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// Required input value was not provided.
    #[error("missing input for ID {0}")]
    MissingInput(u16),

    /// Output references an undefined ID.
    #[error("missing output for ID {0}")]
    MissingOutput(u16),

    /// Internal error: value not computed for ID.
    #[error("undefined value for ID {0}")]
    UndefinedValue(u16),
}
