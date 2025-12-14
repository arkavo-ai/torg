//! DAG representation for TØR-G boolean circuits.

use crate::token::{BoolOp, Source};

/// A node in the boolean circuit.
///
/// Each node performs a binary boolean operation on two source operands.
/// Nodes are stored in topological order, so evaluation can proceed
/// sequentially without dependency resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// The unique identifier for this node.
    pub id: u16,
    /// The boolean operation this node performs.
    pub op: BoolOp,
    /// Left operand source.
    pub left: Source,
    /// Right operand source.
    pub right: Source,
}

impl Node {
    /// Create a new node.
    pub fn new(id: u16, op: BoolOp, left: Source, right: Source) -> Self {
        Self {
            id,
            op,
            left,
            right,
        }
    }
}

/// The complete TØR-G graph (DAG).
///
/// A graph consists of:
/// - Input declarations (external boolean values)
/// - Node definitions (boolean operations on sources)
/// - Output declarations (which node IDs are graph outputs)
///
/// The graph is guaranteed to be a DAG because nodes can only
/// reference previously-defined IDs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    /// Declared input IDs (in declaration order).
    pub inputs: Vec<u16>,
    /// Node definitions (in topological order).
    pub nodes: Vec<Node>,
    /// Output IDs (in declaration order).
    pub outputs: Vec<u16>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of nodes (excluding inputs).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of declared inputs.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of declared outputs.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Calculate graph depth (longest path from input to output).
    ///
    /// Depth is defined as the maximum number of node hops from any
    /// input to any output. A graph with only inputs feeding directly
    /// to outputs has depth 1.
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        // Build depth map: ID -> depth
        // Inputs and constants have depth 0
        // Nodes have depth = 1 + max(depth of sources)
        let mut depths = std::collections::HashMap::new();

        // Initialize inputs with depth 0
        for &id in &self.inputs {
            depths.insert(id, 0usize);
        }

        // Compute depth for each node (nodes are in topological order)
        for node in &self.nodes {
            let left_depth = self.source_depth(&node.left, &depths);
            let right_depth = self.source_depth(&node.right, &depths);
            let node_depth = 1 + left_depth.max(right_depth);
            depths.insert(node.id, node_depth);
        }

        // Return max depth among outputs
        self.outputs
            .iter()
            .filter_map(|id| depths.get(id).copied())
            .max()
            .unwrap_or(0)
    }

    /// Get depth of a source operand.
    fn source_depth(
        &self,
        source: &Source,
        depths: &std::collections::HashMap<u16, usize>,
    ) -> usize {
        match source {
            Source::Id(id) => depths.get(id).copied().unwrap_or(0),
            Source::True | Source::False => 0,
        }
    }

    /// Check if an ID is a declared input.
    pub fn is_input(&self, id: u16) -> bool {
        self.inputs.contains(&id)
    }

    /// Get a node by its ID.
    pub fn get_node(&self, id: u16) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_depth() {
        let graph = Graph::new();
        assert_eq!(graph.depth(), 0);
    }

    #[test]
    fn test_single_node_depth() {
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1))],
            outputs: vec![2],
        };
        assert_eq!(graph.depth(), 1);
    }

    #[test]
    fn test_chain_depth() {
        // input(0) -> node(1) -> node(2) -> output
        let graph = Graph {
            inputs: vec![0],
            nodes: vec![
                Node::new(1, BoolOp::Or, Source::Id(0), Source::True),
                Node::new(2, BoolOp::Or, Source::Id(1), Source::False),
            ],
            outputs: vec![2],
        };
        assert_eq!(graph.depth(), 2);
    }

    #[test]
    fn test_parallel_depth() {
        // Two parallel paths of depth 1
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![
                Node::new(2, BoolOp::Or, Source::Id(0), Source::True),
                Node::new(3, BoolOp::Or, Source::Id(1), Source::False),
            ],
            outputs: vec![2, 3],
        };
        assert_eq!(graph.depth(), 1);
    }
}
