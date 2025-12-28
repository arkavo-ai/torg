//! JSON serialization for TØR-G graphs.
//!
//! Provides human-readable JSON format for debugging and inspection.
//! Uses serde_json with pretty printing enabled.

use torg_core::Graph;

use crate::error::SerdeError;

/// Serialize a graph to pretty-printed JSON.
///
/// This format is intended for debugging and inspection.
/// For production use, prefer the binary format which is more compact.
pub fn to_json(graph: &Graph) -> String {
    serde_json::to_string_pretty(graph).expect("Graph serialization should not fail")
}

/// Serialize a graph to compact JSON (no whitespace).
pub fn to_json_compact(graph: &Graph) -> String {
    serde_json::to_string(graph).expect("Graph serialization should not fail")
}

/// Deserialize a graph from JSON.
pub fn from_json(json: &str) -> Result<Graph, SerdeError> {
    Ok(serde_json::from_str(json)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torg_core::{BoolOp, Node, Source};

    fn sample_graph() -> Graph {
        Graph {
            inputs: vec![0, 1],
            nodes: vec![
                Node::new(2, BoolOp::Or, Source::Id(0), Source::Id(1)),
                Node::new(3, BoolOp::Nor, Source::Id(2), Source::True),
            ],
            outputs: vec![3],
        }
    }

    #[test]
    fn test_roundtrip() {
        let graph = sample_graph();
        let json = to_json(&graph);
        let restored = from_json(&json).unwrap();
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_compact_roundtrip() {
        let graph = sample_graph();
        let json = to_json_compact(&graph);
        let restored = from_json(&json).unwrap();
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::default();
        let json = to_json(&graph);
        let restored = from_json(&json).unwrap();
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_json_is_readable() {
        let graph = sample_graph();
        let json = to_json(&graph);

        // Should contain expected field names
        assert!(json.contains("inputs"));
        assert!(json.contains("nodes"));
        assert!(json.contains("outputs"));
        assert!(json.contains("Or"));
        assert!(json.contains("Nor"));
    }

    #[test]
    fn test_invalid_json() {
        let result = from_json("not valid json");
        assert!(matches!(result, Err(SerdeError::Json(_))));
    }
}
