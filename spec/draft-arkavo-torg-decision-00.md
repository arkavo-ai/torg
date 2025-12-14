# TØR-G: Token-Only Reasoner (Graph)

**Draft:** draft-arkavo-torg-decision-00
**Status:** Draft
**Authors:** Arkavo AI

## Abstract

TØR-G (Token-Only Reasoner - Graph) is a zero-parser, boolean-circuit
intermediate representation (IR) for AI agent policy synthesis. This
document specifies the token vocabulary, graph structure, and evaluation
semantics.

## 1. Introduction

TØR-G enables LLMs to generate formally verifiable boolean circuits by
emitting a constrained token vocabulary. Key properties:

- **No text parsing** — token stream only
- **Deterministic construction** — state machine compilation
- **Pure boolean combinatorics** — no arithmetic, state, or loops
- **DAG structure** — formally verifiable

## 2. Token Vocabulary

### 2.1 Operators

| Token | Symbol | Semantics |
|-------|--------|-----------|
| Or    | ∨      | A ∨ B     |
| Nor   | ⊽      | ¬(A ∨ B)  |
| Xor   | ⊻      | A ⊕ B     |

### 2.2 Structural

| Token      | Symbol | Purpose                    |
|------------|--------|----------------------------|
| NodeStart  | ●      | Begin node definition      |
| NodeEnd    | ○      | End node definition        |
| InputDecl  | ◎IN    | Declare external input     |
| OutputDecl | ◎OUT   | Mark as graph output       |

### 2.3 Literals

| Token | Symbol | Value |
|-------|--------|-------|
| True  | ◎T     | true  |
| False | ◎F     | false |

### 2.4 Identifiers

`Id(u16)` — Unique identifier for inputs and nodes (0..65535)

## 3. Graph Structure

A TØR-G graph consists of:

1. **Inputs**: Declared external boolean values
2. **Nodes**: Binary boolean operations on sources
3. **Outputs**: Which node IDs are graph outputs

Sources can be:
- `Id(n)` — Reference to input or previous node
- `True` — Constant true
- `False` — Constant false

## 4. Construction Grammar

```
graph     = inputs nodes outputs
inputs    = (InputDecl Id)*
nodes     = (NodeStart Id Op Source Source NodeEnd)*
outputs   = (OutputDecl Id)+
Op        = Or | Nor | Xor
Source    = Id | True | False
```

## 5. DAG Constraint

Nodes may only reference IDs that have been previously defined
(inputs or earlier nodes). This ensures the graph is a DAG.

## 6. Example

Policy: "Allow if Admin OR (Owner XOR Public)"

```
InputDecl Id(0)    // is_admin
InputDecl Id(1)    // is_owner
InputDecl Id(2)    // is_public
NodeStart Id(3) Xor Id(1) Id(2) NodeEnd    // owner XOR public
NodeStart Id(4) Or Id(0) Id(3) NodeEnd     // admin OR node(3)
OutputDecl Id(4)
```

## 7. Derived Operations

AND, NOT, NAND, IMPLIES can be derived:

- `NOT(a) = NOR(a, a)`
- `AND(a, b) = NOR(NOR(a, a), NOR(b, b))`
- `NAND(a, b) = NOR(NOR(NOR(a, a), NOR(b, b)), NOR(NOR(a, a), NOR(b, b)))`
- `IMPLIES(a, b) = OR(NOR(a, a), b)`

## 8. Security Considerations

### 8.1 Resource Limits

Implementations MUST enforce limits on:
- Maximum nodes
- Maximum depth
- Maximum inputs
- Maximum outputs

### 8.2 DoS Prevention

The deterministic state machine rejects invalid tokens immediately,
preventing resource exhaustion attacks.

## References

- [Boolean Satisfiability](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)
- [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
