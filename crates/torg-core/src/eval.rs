//! Circuit evaluator for TØR-G graphs.

use std::collections::HashMap;

use crate::error::EvalError;
use crate::graph::Graph;
use crate::token::Source;

/// Evaluate a TØR-G graph with given input values.
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
/// Returns an error if:
/// - A required input is missing
/// - An output references an undefined ID
/// - Internal evaluation fails (should not happen with valid graphs)
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

    // Values cache: stores computed values for all IDs
    let mut values: HashMap<u16, bool> = HashMap::new();

    // Load input values
    for &id in &graph.inputs {
        if let Some(&val) = inputs.get(&id) {
            values.insert(id, val);
        }
    }

    // Evaluate nodes in topological order (already sorted in graph)
    for node in &graph.nodes {
        let left = resolve_source(&node.left, &values)?;
        let right = resolve_source(&node.right, &values)?;
        let result = node.op.eval(left, right);
        values.insert(node.id, result);
    }

    // Collect output values
    let mut outputs = HashMap::new();
    for &id in &graph.outputs {
        let val = values.get(&id).ok_or(EvalError::MissingOutput(id))?;
        outputs.insert(id, *val);
    }

    Ok(outputs)
}

/// Resolve a source operand to its boolean value.
fn resolve_source(source: &Source, values: &HashMap<u16, bool>) -> Result<bool, EvalError> {
    match source {
        Source::True => Ok(true),
        Source::False => Ok(false),
        Source::Id(id) => values
            .get(id)
            .copied()
            .ok_or(EvalError::UndefinedValue(*id)),
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
}
