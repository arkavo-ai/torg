//! Circuit evaluator for TØR-G graphs.
//!
//! Provides high-performance evaluation of boolean circuits using
//! direct array indexing instead of hash maps.

use std::collections::HashMap;

use crate::error::EvalError;
use crate::graph::Graph;
use crate::token::Source;

/// Evaluate a TØR-G graph with given input values (HashMap API).
///
/// This is a convenience wrapper around [`evaluate_into`] for callers
/// that prefer HashMap-based input/output.
///
/// # Arguments
///
/// * `graph` - The graph to evaluate
/// * `inputs` - Map from input IDs to their boolean values
///
/// # Returns
///
/// Map from output IDs to their computed boolean values.
///
/// # Errors
///
/// Returns an error if a required input is missing.
pub fn evaluate(
    graph: &Graph,
    inputs: &HashMap<u16, bool>,
) -> Result<HashMap<u16, bool>, EvalError> {
    // Verify all declared inputs have values
    for &id in &graph.inputs {
        if !inputs.contains_key(&id) {
            return Err(EvalError::MissingInput(id));
        }
    }

    // Convert to slice-based format
    let max_input_id = graph.inputs.iter().copied().max().unwrap_or(0) as usize;
    let mut input_values = vec![false; max_input_id + 1];
    for &id in &graph.inputs {
        input_values[id as usize] = inputs[&id];
    }

    // Evaluate using fast path
    let output_values = evaluate_graph(graph, &input_values);

    // Convert output to HashMap
    let mut result = HashMap::with_capacity(graph.outputs.len());
    for (i, &id) in graph.outputs.iter().enumerate() {
        result.insert(id, output_values[i]);
    }

    Ok(result)
}

/// Evaluate a graph and return output values as a Vec.
///
/// This is the fastest evaluation path. Input values are provided as a slice
/// indexed by input ID. Returns output values in the same order as `graph.outputs`.
///
/// # Arguments
///
/// * `graph` - The graph to evaluate
/// * `inputs` - Slice of boolean values indexed by input ID. Must be large enough
///   to contain all input IDs (i.e., `inputs.len() > max_input_id`).
///
/// # Returns
///
/// Vec of output values in the same order as `graph.outputs`.
///
/// # Panics
///
/// Panics if `inputs` is too small to contain all input IDs.
#[inline]
pub fn evaluate_graph(graph: &Graph, inputs: &[bool]) -> Vec<bool> {
    let mut outputs = vec![false; graph.outputs.len()];
    evaluate_into(graph, inputs, &mut outputs);
    outputs
}

/// Evaluate a graph into a pre-allocated output buffer.
///
/// This is the zero-allocation evaluation path for hot loops.
///
/// # Arguments
///
/// * `graph` - The graph to evaluate
/// * `inputs` - Slice of boolean values indexed by input ID
/// * `outputs` - Pre-allocated output buffer (must have length >= graph.outputs.len())
///
/// # Panics
///
/// Panics if `inputs` or `outputs` are too small.
#[inline]
pub fn evaluate_into(graph: &Graph, inputs: &[bool], outputs: &mut [bool]) {
    // Fast path: empty graph
    if graph.nodes.is_empty() {
        // Outputs are just input references
        for (i, &id) in graph.outputs.iter().enumerate() {
            outputs[i] = inputs[id as usize];
        }
        return;
    }

    // Compute max ID to size the values array
    let max_node_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0);
    let max_input_id = graph.inputs.iter().copied().max().unwrap_or(0);
    let max_id = max_node_id.max(max_input_id) as usize;

    // Pre-allocate values array (indexed by ID)
    let mut values = vec![false; max_id + 1];

    // Load input values
    for &id in &graph.inputs {
        values[id as usize] = inputs[id as usize];
    }

    // Evaluate nodes in topological order
    for node in &graph.nodes {
        let left = resolve_source_fast(&node.left, &values);
        let right = resolve_source_fast(&node.right, &values);
        values[node.id as usize] = node.op.eval(left, right);
    }

    // Write outputs
    for (i, &id) in graph.outputs.iter().enumerate() {
        outputs[i] = values[id as usize];
    }
}

/// Resolve a source operand using direct array indexing.
#[inline(always)]
fn resolve_source_fast(source: &Source, values: &[bool]) -> bool {
    match source {
        Source::True => true,
        Source::False => false,
        Source::Id(id) => values[*id as usize],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;
    use crate::token::BoolOp;

    #[test]
    fn test_simple_or() {
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1))],
            outputs: vec![2],
        };

        // false OR false = false
        let result = evaluate(&graph, &[(0, false), (1, false)].into()).unwrap();
        assert!(!result[&2]);

        // false OR true = true
        let result = evaluate(&graph, &[(0, false), (1, true)].into()).unwrap();
        assert!(result[&2]);

        // true OR false = true
        let result = evaluate(&graph, &[(0, true), (1, false)].into()).unwrap();
        assert!(result[&2]);

        // true OR true = true
        let result = evaluate(&graph, &[(0, true), (1, true)].into()).unwrap();
        assert!(result[&2]);
    }

    #[test]
    fn test_constants() {
        let graph = Graph {
            inputs: vec![],
            nodes: vec![Node::new(0, BoolOp::Or, Source::True, Source::False)],
            outputs: vec![0],
        };

        let result = evaluate(&graph, &HashMap::new()).unwrap();
        assert!(result[&0]); // true OR false = true
    }

    #[test]
    fn test_chain_evaluation() {
        // node(2) = input(0) XOR input(1)
        // node(3) = node(2) OR True
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![
                Node::new(2, BoolOp::Xor, Source::Id(0), Source::Id(1)),
                Node::new(3, BoolOp::Or, Source::Id(2), Source::True),
            ],
            outputs: vec![3],
        };

        // 0 XOR 0 = 0, 0 OR true = true
        let result = evaluate(&graph, &[(0, false), (1, false)].into()).unwrap();
        assert!(result[&3]);
    }

    #[test]
    fn test_missing_input() {
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1))],
            outputs: vec![2],
        };

        // Missing input 1
        let result = evaluate(&graph, &[(0, false)].into());
        assert_eq!(result, Err(EvalError::MissingInput(1)));
    }

    #[test]
    fn test_nor_operation() {
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![Node::new(2, BoolOp::Nor, Source::Id(0), Source::Id(1))],
            outputs: vec![2],
        };

        // false NOR false = true
        let result = evaluate(&graph, &[(0, false), (1, false)].into()).unwrap();
        assert!(result[&2]);

        // Any true = false
        let result = evaluate(&graph, &[(0, true), (1, false)].into()).unwrap();
        assert!(!result[&2]);
    }

    #[test]
    fn test_evaluate_graph_fast() {
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1))],
            outputs: vec![2],
        };

        // Use slice-based API
        let inputs = [false, true]; // id 0 = false, id 1 = true
        let outputs = evaluate_graph(&graph, &inputs);
        assert_eq!(outputs, vec![true]); // false OR true = true
    }

    #[test]
    fn test_evaluate_into_zero_alloc() {
        let graph = Graph {
            inputs: vec![0, 1],
            nodes: vec![
                Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1)),
                Node::new(3, BoolOp::Xor, Source::Id(0), Source::Id(1)),
            ],
            outputs: vec![2, 3],
        };

        let inputs = [true, false];
        let mut outputs = [false, false];
        evaluate_into(&graph, &inputs, &mut outputs);

        assert_eq!(outputs[0], true); // true OR false = true
        assert_eq!(outputs[1], true); // true XOR false = true
    }
}
