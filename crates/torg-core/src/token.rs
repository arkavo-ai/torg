//! Token vocabulary for TØR-G IR.
//!
//! This module defines the closed, finite set of tokens that comprise
//! the TØR-G intermediate representation. LLMs emit these tokens
//! directly, with no text parsing required.

/// TØR-G token vocabulary - closed, finite set.
///
/// These tokens are the atomic units of the IR. An LLM constrained
/// to emit only valid tokens at each step (via logit masking) will
/// produce syntactically correct boolean circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Token {
    // === Logic Operators ===
    /// Logical OR: A ∨ B
    Or,
    /// Logical NOR: ¬(A ∨ B)
    Nor,
    /// Logical XOR: A ⊕ B
    Xor,

    // === Structural ===
    /// Begin node definition: ●
    NodeStart,
    /// End node definition: ○
    NodeEnd,
    /// Declare external input: ◎IN
    InputDecl,
    /// Mark as graph output: ◎OUT
    OutputDecl,

    // === Literals ===
    /// Constant true: ◎T
    True,
    /// Constant false: ◎F
    False,

    // === Identifiers ===
    /// Identifier (0..65535)
    Id(u16),
}

impl Token {
    /// Convert to BoolOp if this is an operator token.
    pub fn as_bool_op(&self) -> Option<BoolOp> {
        match self {
            Token::Or => Some(BoolOp::Or),
            Token::Nor => Some(BoolOp::Nor),
            Token::Xor => Some(BoolOp::Xor),
            _ => None,
        }
    }

    /// Check if this is an operator token.
    pub fn is_operator(&self) -> bool {
        matches!(self, Token::Or | Token::Nor | Token::Xor)
    }

    /// Convert to Source if this can be a source operand.
    pub fn as_source(&self) -> Option<Source> {
        match self {
            Token::Id(id) => Some(Source::Id(*id)),
            Token::True => Some(Source::True),
            Token::False => Some(Source::False),
            _ => None,
        }
    }
}

/// Source operand for a node.
///
/// Represents where a node gets its input value from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// Reference to a declared input or previous node by ID.
    Id(u16),
    /// Constant true.
    True,
    /// Constant false.
    False,
}

/// Binary boolean operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoolOp {
    /// Logical OR: A ∨ B
    Or,
    /// Logical NOR: ¬(A ∨ B)
    Nor,
    /// Logical XOR: A ⊕ B
    Xor,
}

impl BoolOp {
    /// Evaluate this operation on two boolean values.
    #[inline]
    pub fn eval(self, a: bool, b: bool) -> bool {
        match self {
            BoolOp::Or => a || b,
            BoolOp::Nor => !(a || b),
            BoolOp::Xor => a ^ b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_op_or() {
        assert!(!BoolOp::Or.eval(false, false));
        assert!(BoolOp::Or.eval(false, true));
        assert!(BoolOp::Or.eval(true, false));
        assert!(BoolOp::Or.eval(true, true));
    }

    #[test]
    fn test_bool_op_nor() {
        assert!(BoolOp::Nor.eval(false, false));
        assert!(!BoolOp::Nor.eval(false, true));
        assert!(!BoolOp::Nor.eval(true, false));
        assert!(!BoolOp::Nor.eval(true, true));
    }

    #[test]
    fn test_bool_op_xor() {
        assert!(!BoolOp::Xor.eval(false, false));
        assert!(BoolOp::Xor.eval(false, true));
        assert!(BoolOp::Xor.eval(true, false));
        assert!(!BoolOp::Xor.eval(true, true));
    }

    #[test]
    fn test_token_as_bool_op() {
        assert_eq!(Token::Or.as_bool_op(), Some(BoolOp::Or));
        assert_eq!(Token::Nor.as_bool_op(), Some(BoolOp::Nor));
        assert_eq!(Token::Xor.as_bool_op(), Some(BoolOp::Xor));
        assert_eq!(Token::NodeStart.as_bool_op(), None);
        assert_eq!(Token::Id(0).as_bool_op(), None);
    }

    #[test]
    fn test_token_as_source() {
        assert_eq!(Token::Id(42).as_source(), Some(Source::Id(42)));
        assert_eq!(Token::True.as_source(), Some(Source::True));
        assert_eq!(Token::False.as_source(), Some(Source::False));
        assert_eq!(Token::Or.as_source(), None);
        assert_eq!(Token::NodeStart.as_source(), None);
    }
}
