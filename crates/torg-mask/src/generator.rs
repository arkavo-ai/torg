//! Mask generator for creating logit masks from builder state.

use torg_core::{Builder, Token};

use crate::mapping::TokenMapping;
use crate::mask::LogitMask;

/// Generates logit masks from TØR-G builder state.
///
/// This struct combines a token mapping with a vocabulary size to
/// produce `LogitMask` instances from `Builder::valid_next_tokens()`.
#[derive(Debug, Clone)]
pub struct MaskGenerator {
    mapping: TokenMapping,
    vocab_size: usize,
}

impl MaskGenerator {
    /// Create a new mask generator.
    ///
    /// # Arguments
    ///
    /// * `mapping` - Token mapping from TØR-G tokens to LLM vocab IDs
    /// * `vocab_size` - Size of the LLM vocabulary
    pub fn new(mapping: TokenMapping, vocab_size: usize) -> Self {
        Self {
            mapping,
            vocab_size,
        }
    }

    /// Get the token mapping.
    pub fn mapping(&self) -> &TokenMapping {
        &self.mapping
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Generate a logit mask from a list of valid tokens.
    ///
    /// Tokens that cannot be mapped are silently ignored.
    pub fn generate(&self, valid_tokens: &[Token]) -> LogitMask {
        let allowed: Vec<u32> = valid_tokens
            .iter()
            .filter_map(|&token| self.mapping.get(token))
            .collect();

        LogitMask::new(self.vocab_size, allowed)
    }

    /// Generate a logit mask from builder state.
    ///
    /// This is a convenience method that calls `builder.valid_next_tokens()`
    /// and then generates the mask.
    pub fn generate_from_builder(&self, builder: &Builder) -> LogitMask {
        self.generate(&builder.valid_next_tokens())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torg_core::Token;

    #[test]
    fn test_generate_operators() {
        let mapping = TokenMapping::sequential(256);
        let generator = MaskGenerator::new(mapping, 1000);

        let valid = vec![Token::Or, Token::Nor, Token::Xor];
        let mask = generator.generate(&valid);

        assert_eq!(mask.allowed_count(), 3);
        assert!(mask.is_allowed(0)); // Or
        assert!(mask.is_allowed(1)); // Nor
        assert!(mask.is_allowed(2)); // Xor
        assert!(!mask.is_allowed(3)); // NodeStart - not allowed
    }

    #[test]
    fn test_generate_with_ids() {
        let mapping = TokenMapping::sequential(256);
        let generator = MaskGenerator::new(mapping, 1000);

        let valid = vec![Token::Id(0), Token::Id(1), Token::True, Token::False];
        let mask = generator.generate(&valid);

        assert_eq!(mask.allowed_count(), 4);
        assert!(mask.is_allowed(9)); // Id(0)
        assert!(mask.is_allowed(10)); // Id(1)
        assert!(mask.is_allowed(7)); // True
        assert!(mask.is_allowed(8)); // False
    }

    #[test]
    fn test_generate_from_builder() {
        let mapping = TokenMapping::sequential(256);
        let generator = MaskGenerator::new(mapping, 1000);

        let mut builder = Builder::new();
        builder.push(Token::InputDecl).unwrap();
        builder.push(Token::Id(0)).unwrap();
        builder.push(Token::NodeStart).unwrap();
        builder.push(Token::Id(1)).unwrap();

        // After node ID, only operators are valid
        let mask = generator.generate_from_builder(&builder);

        assert_eq!(mask.allowed_count(), 3);
        assert!(mask.is_allowed(0)); // Or
        assert!(mask.is_allowed(1)); // Nor
        assert!(mask.is_allowed(2)); // Xor
    }

    #[test]
    fn test_unmapped_tokens_ignored() {
        let mapping = TokenMapping::sequential(10); // Only Id(0)..Id(9)
        let generator = MaskGenerator::new(mapping, 1000);

        let valid = vec![Token::Id(0), Token::Id(100)]; // Id(100) out of range
        let mask = generator.generate(&valid);

        assert_eq!(mask.allowed_count(), 1);
        assert!(mask.is_allowed(9)); // Id(0)
    }
}
