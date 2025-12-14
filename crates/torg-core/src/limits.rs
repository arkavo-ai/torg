//! Resource limits for DoS prevention.

/// Resource limits for graph construction.
///
/// These limits prevent excessive resource consumption during
/// graph construction, providing DoS protection when processing
/// untrusted token streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum number of nodes in the graph.
    pub max_nodes: usize,
    /// Maximum depth of the DAG (longest path from input to output).
    pub max_depth: usize,
    /// Maximum number of input declarations.
    pub max_inputs: usize,
    /// Maximum number of output declarations.
    pub max_outputs: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_nodes: 256,
            max_depth: 64,
            max_inputs: 128,
            max_outputs: 32,
        }
    }
}

impl Limits {
    /// Create limits with custom values.
    pub fn new(max_nodes: usize, max_depth: usize, max_inputs: usize, max_outputs: usize) -> Self {
        Self {
            max_nodes,
            max_depth,
            max_inputs,
            max_outputs,
        }
    }

    /// Permissive limits for testing.
    pub fn permissive() -> Self {
        Self {
            max_nodes: 4096,
            max_depth: 256,
            max_inputs: 1024,
            max_outputs: 256,
        }
    }

    /// Strict limits for production LLM use.
    pub fn strict() -> Self {
        Self {
            max_nodes: 64,
            max_depth: 16,
            max_inputs: 32,
            max_outputs: 8,
        }
    }
}
