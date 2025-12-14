//! Golden vector integration tests.
//!
//! These tests verify the complete pipeline from token stream to evaluation.

use std::collections::HashMap;
use torg_core::{evaluate, Builder, Token};

/// Build a graph from a token stream.
fn build_graph(tokens: &[Token]) -> torg_core::Graph {
    let mut builder = Builder::new();
    for &token in tokens {
        builder.push(token).expect("token should be accepted");
    }
    builder.finish().expect("graph should be valid")
}

/// Test case: Allow if Admin OR (Owner XOR Public)
///
/// This is the canonical example from the spec.
///
/// Inputs:
/// - 0: is_admin
/// - 1: is_owner
/// - 2: is_public
///
/// Nodes:
/// - 3: owner XOR public
/// - 4: admin OR node(3)
///
/// Output: node(4)
#[test]
fn test_admin_or_owner_xor_public() {
    use Token::*;

    let tokens = vec![
        // Declare inputs
        InputDecl,
        Id(0), // is_admin
        InputDecl,
        Id(1), // is_owner
        InputDecl,
        Id(2), // is_public
        // Node 3: owner XOR public
        NodeStart,
        Id(3),
        Xor,
        Id(1),
        Id(2),
        NodeEnd,
        // Node 4: admin OR node(3)
        NodeStart,
        Id(4),
        Or,
        Id(0),
        Id(3),
        NodeEnd,
        // Output
        OutputDecl,
        Id(4),
    ];

    let graph = build_graph(&tokens);

    // Verify graph structure
    assert_eq!(graph.inputs, vec![0, 1, 2]);
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.outputs, vec![4]);

    // Test all 8 combinations
    // (is_admin, is_owner, is_public) -> expected
    let test_cases = [
        ((false, false, false), false), // nothing -> deny
        ((false, false, true), true),   // public only -> XOR true -> allow
        ((false, true, false), true),   // owner only -> XOR true -> allow
        ((false, true, true), false),   // owner AND public -> XOR false -> deny
        ((true, false, false), true),   // admin only -> allow
        ((true, false, true), true),    // admin + public -> allow
        ((true, true, false), true),    // admin + owner -> allow
        ((true, true, true), true),     // all three -> admin overrides -> allow
    ];

    for ((admin, owner, public), expected) in test_cases {
        let inputs: HashMap<u16, bool> = [(0, admin), (1, owner), (2, public)].into();

        let outputs = evaluate(&graph, &inputs).unwrap();

        assert_eq!(
            outputs[&4], expected,
            "Failed for admin={}, owner={}, public={}",
            admin, owner, public
        );
    }
}

/// Test: Simple AND using NOR gates
///
/// AND(a, b) = NOR(NOR(a, a), NOR(b, b))
/// = NOR(NOT(a), NOT(b))
///
/// But since we only have binary NOR:
/// NOT(a) = NOR(a, a)
/// AND(a, b) = NOR(NOR(a,a), NOR(b,b))
#[test]
fn test_and_via_nor() {
    use Token::*;

    let tokens = vec![
        InputDecl,
        Id(0), // a
        InputDecl,
        Id(1), // b
        // Node 2: NOR(a, a) = NOT(a)
        NodeStart,
        Id(2),
        Nor,
        Id(0),
        Id(0),
        NodeEnd,
        // Node 3: NOR(b, b) = NOT(b)
        NodeStart,
        Id(3),
        Nor,
        Id(1),
        Id(1),
        NodeEnd,
        // Node 4: NOR(NOT(a), NOT(b)) = AND(a, b)
        NodeStart,
        Id(4),
        Nor,
        Id(2),
        Id(3),
        NodeEnd,
        OutputDecl,
        Id(4),
    ];

    let graph = build_graph(&tokens);

    // Test AND truth table
    let test_cases = [
        ((false, false), false),
        ((false, true), false),
        ((true, false), false),
        ((true, true), true),
    ];

    for ((a, b), expected) in test_cases {
        let inputs: HashMap<u16, bool> = [(0, a), (1, b)].into();
        let outputs = evaluate(&graph, &inputs).unwrap();
        assert_eq!(outputs[&4], expected, "Failed for a={}, b={}", a, b);
    }
}

/// Test: NOT using NOR
///
/// NOT(a) = NOR(a, a)
#[test]
fn test_not_via_nor() {
    use Token::*;

    let tokens = vec![
        InputDecl,
        Id(0),
        NodeStart,
        Id(1),
        Nor,
        Id(0),
        Id(0),
        NodeEnd,
        OutputDecl,
        Id(1),
    ];

    let graph = build_graph(&tokens);

    let inputs_false: HashMap<u16, bool> = [(0, false)].into();
    let inputs_true: HashMap<u16, bool> = [(0, true)].into();

    assert!(evaluate(&graph, &inputs_false).unwrap()[&1]);
    assert!(!evaluate(&graph, &inputs_true).unwrap()[&1]);
}

/// Test: Constants only (no inputs)
#[test]
fn test_constants_only() {
    use Token::*;

    let tokens = vec![
        NodeStart,
        Id(0),
        Xor,
        True,
        False,
        NodeEnd,
        OutputDecl,
        Id(0),
    ];

    let graph = build_graph(&tokens);
    assert!(graph.inputs.is_empty());

    let outputs = evaluate(&graph, &HashMap::new()).unwrap();
    assert!(outputs[&0]); // true XOR false = true
}

/// Test: Multiple outputs
#[test]
fn test_multiple_outputs() {
    use Token::*;

    let tokens = vec![
        InputDecl,
        Id(0),
        InputDecl,
        Id(1),
        // Node 2: OR
        NodeStart,
        Id(2),
        Or,
        Id(0),
        Id(1),
        NodeEnd,
        // Node 3: XOR
        NodeStart,
        Id(3),
        Xor,
        Id(0),
        Id(1),
        NodeEnd,
        // Node 4: NOR
        NodeStart,
        Id(4),
        Nor,
        Id(0),
        Id(1),
        NodeEnd,
        // All three as outputs
        OutputDecl,
        Id(2),
        OutputDecl,
        Id(3),
        OutputDecl,
        Id(4),
    ];

    let graph = build_graph(&tokens);
    assert_eq!(graph.outputs, vec![2, 3, 4]);

    let inputs: HashMap<u16, bool> = [(0, true), (1, false)].into();
    let outputs = evaluate(&graph, &inputs).unwrap();

    assert!(outputs[&2]); // true OR false = true
    assert!(outputs[&3]); // true XOR false = true
    assert!(!outputs[&4]); // true NOR false = false
}

/// Test: Deep chain (stress test for depth calculation)
#[test]
fn test_deep_chain() {
    use Token::*;

    let mut tokens = vec![InputDecl, Id(0)];

    // Create a chain of 10 OR nodes
    for i in 1..=10u16 {
        tokens.extend([NodeStart, Id(i), Or, Id(i - 1), True, NodeEnd]);
    }

    tokens.extend([OutputDecl, Id(10)]);

    let graph = build_graph(&tokens);
    assert_eq!(graph.depth(), 10);

    let inputs: HashMap<u16, bool> = [(0, false)].into();
    let outputs = evaluate(&graph, &inputs).unwrap();
    assert!(outputs[&10]); // Chain of ORs with True = true
}

/// Test: Input used as output directly
/// This should work - an input ID can be declared as an output.
#[test]
fn test_input_as_output() {
    use Token::*;

    // We need at least one node, but we can output an input
    // Actually, looking at the code, outputs must reference defined IDs
    // and inputs are defined. Let's verify this works.
    let tokens = vec![
        InputDecl,
        Id(0),
        // Need at least one node for a valid graph? Let's check...
        // Actually no, but we need an output that references a defined ID
        OutputDecl,
        Id(0), // Output the input directly
    ];

    let graph = build_graph(&tokens);
    assert_eq!(graph.inputs, vec![0]);
    assert_eq!(graph.outputs, vec![0]);
    assert!(graph.nodes.is_empty());

    let inputs: HashMap<u16, bool> = [(0, true)].into();
    let outputs = evaluate(&graph, &inputs).unwrap();
    assert!(outputs[&0]);
}

/// Test: valid_next_tokens produces valid sequences
#[test]
fn test_valid_next_tokens_guidance() {
    use Token::*;

    let mut builder = Builder::new();

    // Initially, should accept InputDecl, NodeStart, or OutputDecl
    let valid = builder.valid_next_tokens();
    assert!(valid.contains(&InputDecl));
    assert!(valid.contains(&NodeStart));

    // After InputDecl, should accept Id
    builder.push(InputDecl).unwrap();
    let valid = builder.valid_next_tokens();
    assert!(valid.contains(&Id(0)));
    assert!(valid.contains(&Id(1)));

    builder.push(Id(0)).unwrap();

    // After input declaration, should accept InputDecl, NodeStart, OutputDecl
    let valid = builder.valid_next_tokens();
    assert!(valid.contains(&InputDecl));
    assert!(valid.contains(&NodeStart));

    // Start a node
    builder.push(NodeStart).unwrap();
    let valid = builder.valid_next_tokens();
    // Should only accept unused IDs
    assert!(valid.contains(&Id(1))); // 1 not defined
    assert!(!valid.contains(&Id(0))); // 0 already defined

    builder.push(Id(1)).unwrap();

    // Should accept only operators
    let valid = builder.valid_next_tokens();
    assert_eq!(valid.len(), 3);
    assert!(valid.contains(&Or));
    assert!(valid.contains(&Nor));
    assert!(valid.contains(&Xor));
}
