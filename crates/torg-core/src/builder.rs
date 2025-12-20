//! State machine builder for TØR-G graph construction.
//!
//! The builder consumes tokens one-by-one and constructs the graph,
//! enforcing all structural constraints. It provides `valid_next_tokens()`
//! for LLM logit masking.

use std::collections::HashSet;

use crate::error::BuildError;
use crate::graph::{Graph, Node};
use crate::limits::Limits;
use crate::token::{BoolOp, Source, Token};

/// Builder state machine phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Accepting input declarations.
    Inputs,
    /// Accepting node definitions or output declarations.
    Nodes,
    /// Accepting only output declarations.
    Outputs,
    /// Graph complete, no more tokens accepted.
    Done,
}

/// State within the Inputs phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum InputState {
    /// Ready for InputDecl, NodeStart, or OutputDecl.
    #[default]
    Ready,
    /// Just saw InputDecl, expecting Id.
    ExpectId,
}

/// State within the Outputs phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum OutputState {
    /// Ready for OutputDecl or end.
    #[default]
    Ready,
    /// Just saw OutputDecl, expecting Id.
    ExpectId,
}

/// State within a node definition (● ... ○).
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    /// Expecting node ID after NodeStart.
    ExpectId,
    /// Expecting operator after node ID.
    ExpectOp { id: u16 },
    /// Expecting first source operand.
    ExpectLeft { id: u16, op: BoolOp },
    /// Expecting second source operand.
    ExpectRight { id: u16, op: BoolOp, left: Source },
    /// Expecting NodeEnd.
    ExpectEnd {
        id: u16,
        op: BoolOp,
        left: Source,
        right: Source,
    },
}

/// Deterministic state machine for graph construction.
///
/// The builder enforces all structural constraints:
/// - No duplicate ID definitions
/// - No forward references (DAG property)
/// - Resource limits (max nodes, depth, inputs, outputs)
#[derive(Debug)]
pub struct Builder {
    graph: Graph,
    phase: Phase,
    node_state: Option<NodeState>,
    input_state: InputState,
    output_state: OutputState,
    /// All defined IDs (inputs + nodes).
    defined: HashSet<u16>,
    limits: Limits,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    /// Create a new builder with default limits.
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    /// Create a new builder with custom limits.
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            graph: Graph::new(),
            phase: Phase::Inputs,
            node_state: None,
            input_state: InputState::default(),
            output_state: OutputState::default(),
            defined: HashSet::new(),
            limits,
        }
    }

    /// Get the current build phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Check if the graph is in a completable state.
    ///
    /// Returns `true` when:
    /// - At least one output has been declared
    /// - No node definition is in progress
    /// - We're ready for another output (not mid-declaration)
    ///
    /// This is useful for LLM decoding where we want to know if
    /// generation can stop, even though more outputs could be added.
    pub fn is_completable(&self) -> bool {
        !self.graph.outputs.is_empty()
            && self.node_state.is_none()
            && self.output_state == OutputState::Ready
    }

    /// Feed a single token to the builder.
    pub fn push(&mut self, token: Token) -> Result<(), BuildError> {
        match self.phase {
            Phase::Done => Err(BuildError::UnexpectedToken),
            Phase::Inputs => self.handle_inputs(token),
            Phase::Nodes => self.handle_nodes(token),
            Phase::Outputs => self.handle_outputs(token),
        }
    }

    /// Finalize and return the graph.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A node definition is incomplete
    /// - No outputs have been declared
    pub fn finish(self) -> Result<Graph, BuildError> {
        if self.node_state.is_some() {
            return Err(BuildError::IncompleteNode);
        }
        if self.graph.outputs.is_empty() {
            return Err(BuildError::NoOutputs);
        }
        Ok(self.graph)
    }

    /// Handle tokens in the Inputs phase.
    fn handle_inputs(&mut self, token: Token) -> Result<(), BuildError> {
        match self.input_state {
            InputState::Ready => {
                // Expecting InputDecl, NodeStart, or OutputDecl
                match token {
                    Token::InputDecl => {
                        // Check limits before accepting
                        if self.graph.inputs.len() >= self.limits.max_inputs {
                            return Err(BuildError::MaxInputsExceeded(self.limits.max_inputs));
                        }
                        self.input_state = InputState::ExpectId;
                        Ok(())
                    }
                    Token::NodeStart => {
                        // Check limits
                        if self.graph.nodes.len() >= self.limits.max_nodes {
                            return Err(BuildError::MaxNodesExceeded(self.limits.max_nodes));
                        }
                        // Transition to Nodes phase
                        self.phase = Phase::Nodes;
                        self.node_state = Some(NodeState::ExpectId);
                        Ok(())
                    }
                    Token::OutputDecl => {
                        // Skip directly to Outputs phase (empty nodes section)
                        self.phase = Phase::Outputs;
                        self.output_state = OutputState::ExpectId;
                        Ok(())
                    }
                    _ => Err(BuildError::UnexpectedToken),
                }
            }
            InputState::ExpectId => {
                // Expecting an Id after InputDecl
                if let Token::Id(id) = token {
                    // Check for duplicate
                    if self.defined.contains(&id) {
                        return Err(BuildError::DuplicateDefinition(id));
                    }
                    self.defined.insert(id);
                    self.graph.inputs.push(id);
                    self.input_state = InputState::Ready;
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
        }
    }

    /// Handle tokens in the Nodes phase.
    fn handle_nodes(&mut self, token: Token) -> Result<(), BuildError> {
        match self.node_state {
            None => {
                // Between nodes, expecting NodeStart or OutputDecl
                match token {
                    Token::NodeStart => {
                        // Check limits
                        if self.graph.nodes.len() >= self.limits.max_nodes {
                            return Err(BuildError::MaxNodesExceeded(self.limits.max_nodes));
                        }
                        self.node_state = Some(NodeState::ExpectId);
                        Ok(())
                    }
                    Token::OutputDecl => {
                        // Check limits before accepting
                        if self.graph.outputs.len() >= self.limits.max_outputs {
                            return Err(BuildError::MaxOutputsExceeded(self.limits.max_outputs));
                        }
                        self.phase = Phase::Outputs;
                        self.output_state = OutputState::ExpectId;
                        Ok(())
                    }
                    _ => Err(BuildError::UnexpectedToken),
                }
            }
            Some(state) => self.handle_node_state(state, token),
        }
    }

    /// Handle tokens within a node definition.
    fn handle_node_state(&mut self, state: NodeState, token: Token) -> Result<(), BuildError> {
        match state {
            NodeState::ExpectId => {
                if let Token::Id(id) = token {
                    // Check for duplicate
                    if self.defined.contains(&id) {
                        return Err(BuildError::DuplicateDefinition(id));
                    }
                    self.node_state = Some(NodeState::ExpectOp { id });
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
            NodeState::ExpectOp { id } => {
                if let Some(op) = token.as_bool_op() {
                    self.node_state = Some(NodeState::ExpectLeft { id, op });
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
            NodeState::ExpectLeft { id, op } => {
                if let Some(source) = token.as_source() {
                    // Validate source reference
                    if let Source::Id(ref_id) = source {
                        if !self.defined.contains(&ref_id) {
                            return Err(BuildError::UndefinedReference(ref_id));
                        }
                    }
                    self.node_state = Some(NodeState::ExpectRight {
                        id,
                        op,
                        left: source,
                    });
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
            NodeState::ExpectRight { id, op, left } => {
                if let Some(source) = token.as_source() {
                    // Validate source reference
                    if let Source::Id(ref_id) = source {
                        if !self.defined.contains(&ref_id) {
                            return Err(BuildError::UndefinedReference(ref_id));
                        }
                    }
                    self.node_state = Some(NodeState::ExpectEnd {
                        id,
                        op,
                        left,
                        right: source,
                    });
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
            NodeState::ExpectEnd {
                id,
                op,
                left,
                right,
            } => {
                if token == Token::NodeEnd {
                    // Commit the node
                    let node = Node::new(id, op, left, right);
                    self.defined.insert(id);
                    self.graph.nodes.push(node);

                    // Check depth limit
                    let depth = self.graph.depth();
                    if depth > self.limits.max_depth {
                        return Err(BuildError::MaxDepthExceeded(self.limits.max_depth));
                    }

                    self.node_state = None;
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
        }
    }

    /// Handle tokens in the Outputs phase.
    fn handle_outputs(&mut self, token: Token) -> Result<(), BuildError> {
        match self.output_state {
            OutputState::Ready => {
                // Expecting OutputDecl
                match token {
                    Token::OutputDecl => {
                        // Check limits before accepting
                        if self.graph.outputs.len() >= self.limits.max_outputs {
                            return Err(BuildError::MaxOutputsExceeded(self.limits.max_outputs));
                        }
                        self.output_state = OutputState::ExpectId;
                        Ok(())
                    }
                    _ => {
                        self.phase = Phase::Done;
                        Err(BuildError::UnexpectedToken)
                    }
                }
            }
            OutputState::ExpectId => {
                // Expecting an Id after OutputDecl
                if let Token::Id(id) = token {
                    // Output must reference a defined ID
                    if !self.defined.contains(&id) {
                        return Err(BuildError::UndefinedReference(id));
                    }
                    self.graph.outputs.push(id);
                    self.output_state = OutputState::Ready;
                    Ok(())
                } else {
                    Err(BuildError::UnexpectedToken)
                }
            }
        }
    }

    /// Returns the set of valid next tokens for LLM logit masking.
    ///
    /// This is the critical interface for constraining LLM output.
    /// By masking all logits except those corresponding to valid tokens,
    /// the LLM is guaranteed to produce a syntactically correct circuit.
    pub fn valid_next_tokens(&self) -> Vec<Token> {
        match self.phase {
            Phase::Done => vec![],
            Phase::Inputs => self.valid_in_inputs(),
            Phase::Nodes => self.valid_in_nodes(),
            Phase::Outputs => self.valid_in_outputs(),
        }
    }

    fn valid_in_inputs(&self) -> Vec<Token> {
        match self.input_state {
            InputState::Ready => {
                // Ready for structural tokens, NOT Id
                let mut valid = Vec::new();

                // Can declare more inputs (up to limit)
                if self.graph.inputs.len() < self.limits.max_inputs {
                    valid.push(Token::InputDecl);
                }

                // Can start defining nodes
                if self.graph.nodes.len() < self.limits.max_nodes {
                    valid.push(Token::NodeStart);
                }

                // Can go directly to outputs
                if self.graph.outputs.len() < self.limits.max_outputs {
                    valid.push(Token::OutputDecl);
                }

                valid
            }
            InputState::ExpectId => {
                // After InputDecl, only Id tokens are valid
                let mut valid = Vec::new();
                for id in 0..=255u16 {
                    if !self.defined.contains(&id) {
                        valid.push(Token::Id(id));
                    }
                }
                valid
            }
        }
    }

    fn valid_in_nodes(&self) -> Vec<Token> {
        match &self.node_state {
            None => {
                // Between nodes
                let mut valid = Vec::new();
                if self.graph.nodes.len() < self.limits.max_nodes {
                    valid.push(Token::NodeStart);
                }
                if self.graph.outputs.len() < self.limits.max_outputs
                    && !self.graph.nodes.is_empty()
                {
                    valid.push(Token::OutputDecl);
                }
                valid
            }
            Some(state) => self.valid_in_node_state(state),
        }
    }

    fn valid_in_node_state(&self, state: &NodeState) -> Vec<Token> {
        match state {
            NodeState::ExpectId => {
                // Any ID not yet defined
                let mut valid = Vec::new();
                for id in 0..=255u16 {
                    if !self.defined.contains(&id) {
                        valid.push(Token::Id(id));
                    }
                }
                valid
            }
            NodeState::ExpectOp { .. } => {
                vec![Token::Or, Token::Nor, Token::Xor]
            }
            NodeState::ExpectLeft { .. } | NodeState::ExpectRight { .. } => {
                // Can reference any defined ID, or use constants
                let mut valid = Vec::new();
                for &id in &self.defined {
                    valid.push(Token::Id(id));
                }
                valid.push(Token::True);
                valid.push(Token::False);
                valid
            }
            NodeState::ExpectEnd { .. } => {
                vec![Token::NodeEnd]
            }
        }
    }

    fn valid_in_outputs(&self) -> Vec<Token> {
        match self.output_state {
            OutputState::Ready => {
                // Ready for OutputDecl only (or end of graph)
                let mut valid = Vec::new();
                if self.graph.outputs.len() < self.limits.max_outputs {
                    valid.push(Token::OutputDecl);
                }
                valid
            }
            OutputState::ExpectId => {
                // After OutputDecl, only defined IDs are valid
                let mut valid = Vec::new();
                for &id in &self.defined {
                    valid.push(Token::Id(id));
                }
                valid
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_build() {
        let mut builder = Builder::new();

        // Declare inputs
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(1)).unwrap();

        // Define a node: node(2) = input(0) OR input(1)
        builder.push(Token::NodeStart).unwrap();
        builder.push(Token::Id(2)).unwrap();
        builder.push(Token::Or).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::Id(1)).unwrap();
        builder.push(Token::NodeEnd).unwrap();

        // Declare output
        builder.push(Token::OutputDecl).unwrap();
        builder.push(Token::Id(2)).unwrap();

        let graph = builder.finish().unwrap();
        assert_eq!(graph.inputs, vec![0, 1]);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.outputs, vec![2]);
    }

    #[test]
    fn test_duplicate_definition() {
        let mut builder = Builder::new();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::InputDecl).unwrap();
        assert_eq!(
            builder.push(Token::Id(0)),
            Err(BuildError::DuplicateDefinition(0))
        );
    }

    #[test]
    fn test_undefined_reference() {
        let mut builder = Builder::new();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::NodeStart).unwrap();
        builder.push(Token::Id(1)).unwrap();
        builder.push(Token::Or).unwrap();
        builder.push(Token::Id(0)).unwrap();
        // Reference to undefined ID 99
        assert_eq!(
            builder.push(Token::Id(99)),
            Err(BuildError::UndefinedReference(99))
        );
    }

    #[test]
    fn test_no_outputs_error() {
        let mut builder = Builder::new();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        assert_eq!(builder.finish(), Err(BuildError::NoOutputs));
    }

    #[test]
    fn test_incomplete_node_error() {
        let mut builder = Builder::new();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::NodeStart).unwrap();
        builder.push(Token::Id(1)).unwrap();
        builder.push(Token::Or).unwrap();
        // Node incomplete
        assert_eq!(builder.finish(), Err(BuildError::IncompleteNode));
    }

    #[test]
    fn test_valid_next_tokens_initial() {
        let builder = Builder::new();
        let valid = builder.valid_next_tokens();
        assert!(valid.contains(&Token::InputDecl));
        assert!(valid.contains(&Token::NodeStart));
        assert!(valid.contains(&Token::OutputDecl));
    }

    #[test]
    fn test_valid_next_tokens_expect_op() {
        let mut builder = Builder::new();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::NodeStart).unwrap();
        builder.push(Token::Id(1)).unwrap();

        let valid = builder.valid_next_tokens();
        assert_eq!(valid, vec![Token::Or, Token::Nor, Token::Xor]);
    }

    #[test]
    fn test_constants_as_sources() {
        let mut builder = Builder::new();

        // No inputs, just use constants
        builder.push(Token::NodeStart).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::Or).unwrap();
        builder.push(Token::True).unwrap();
        builder.push(Token::False).unwrap();
        builder.push(Token::NodeEnd).unwrap();

        builder.push(Token::OutputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();

        let graph = builder.finish().unwrap();
        assert!(graph.inputs.is_empty());
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].left, Source::True);
        assert_eq!(graph.nodes[0].right, Source::False);
    }
}
