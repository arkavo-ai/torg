//! High-level constrained decoding orchestrator.

use torg_core::{BuildError, Builder, Graph, Limits, Phase};

use crate::generator::MaskGenerator;
use crate::mask::LogitMask;

/// High-level orchestrator for constrained LLM decoding.
///
/// This struct manages the decode loop for generating TØR-G graphs
/// from an LLM. It maintains builder state, generates masks, and
/// converts LLM tokens back to TØR-G tokens.
///
/// # Example
///
/// ```ignore
/// let mapping = TokenMapping::sequential(256);
/// let generator = MaskGenerator::new(mapping, vocab_size);
/// let mut decoder = ConstrainedDecoder::new(generator);
///
/// while !decoder.is_complete() {
///     let mask = decoder.next_mask();
///     mask.apply_to_logits(&mut logits);
///     let token_id = sample(&logits);
///     decoder.feed_token(token_id)?;
/// }
///
/// let graph = decoder.finish()?;
/// ```
#[derive(Debug)]
pub struct ConstrainedDecoder {
    builder: Builder,
    generator: MaskGenerator,
}

impl ConstrainedDecoder {
    /// Create a new constrained decoder with default limits.
    pub fn new(generator: MaskGenerator) -> Self {
        Self {
            builder: Builder::new(),
            generator,
        }
    }

    /// Create a new constrained decoder with custom limits.
    pub fn with_limits(generator: MaskGenerator, limits: Limits) -> Self {
        Self {
            builder: Builder::with_limits(limits),
            generator,
        }
    }

    /// Get the current build phase.
    pub fn phase(&self) -> Phase {
        self.builder.phase()
    }

    /// Check if decoding is complete.
    ///
    /// Returns `true` when the builder is in the `Done` phase or when
    /// there are no more valid tokens (which shouldn't happen in normal use).
    pub fn is_complete(&self) -> bool {
        self.builder.phase() == Phase::Done
    }

    /// Generate the logit mask for the next token.
    ///
    /// Apply this mask to your LLM's logits before sampling.
    pub fn next_mask(&self) -> LogitMask {
        self.generator.generate_from_builder(&self.builder)
    }

    /// Feed an LLM token ID into the decoder.
    ///
    /// The token ID is reverse-mapped to a TØR-G token and pushed
    /// to the internal builder.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The token ID doesn't map to a valid TØR-G token
    /// - The token is not valid in the current builder state
    pub fn feed_token(&mut self, llm_token_id: u32) -> Result<(), DecodeError> {
        let token = self
            .generator
            .mapping()
            .reverse(llm_token_id)
            .ok_or(DecodeError::UnmappedToken(llm_token_id))?;

        self.builder.push(token).map_err(DecodeError::BuildError)?;

        Ok(())
    }

    /// Finish decoding and return the constructed graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is incomplete (e.g., no outputs declared).
    pub fn finish(self) -> Result<Graph, BuildError> {
        self.builder.finish()
    }

    /// Get a reference to the internal builder.
    pub fn builder(&self) -> &Builder {
        &self.builder
    }

    /// Get the mask generator.
    pub fn generator(&self) -> &MaskGenerator {
        &self.generator
    }
}

/// Errors that can occur during constrained decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The LLM token ID doesn't map to any TØR-G token.
    UnmappedToken(u32),
    /// The TØR-G token is not valid in the current state.
    BuildError(BuildError),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnmappedToken(id) => write!(f, "unmapped LLM token ID: {}", id),
            DecodeError::BuildError(e) => write!(f, "build error: {}", e),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::BuildError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<BuildError> for DecodeError {
    fn from(e: BuildError) -> Self {
        DecodeError::BuildError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::TokenMapping;
    use torg_core::Token;

    fn make_decoder() -> ConstrainedDecoder {
        let mapping = TokenMapping::sequential(256);
        let generator = MaskGenerator::new(mapping, 1000);
        ConstrainedDecoder::new(generator)
    }

    #[test]
    fn test_simple_decode() {
        let mut decoder = make_decoder();

        // Build: InputDecl Id(0) NodeStart Id(1) Or Id(0) True NodeEnd OutputDecl Id(1)
        let token_ids = [
            5,  // InputDecl
            9,  // Id(0)
            3,  // NodeStart
            10, // Id(1)
            0,  // Or
            9,  // Id(0)
            7,  // True
            4,  // NodeEnd
            6,  // OutputDecl
            10, // Id(1)
        ];

        for &id in &token_ids {
            assert!(!decoder.is_complete());
            let mask = decoder.next_mask();
            assert!(mask.is_allowed(id), "token {} should be allowed", id);
            decoder.feed_token(id).unwrap();
        }

        let graph = decoder.finish().unwrap();
        assert_eq!(graph.inputs, vec![0]);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.outputs, vec![1]);
    }

    #[test]
    fn test_unmapped_token_error() {
        let mut decoder = make_decoder();
        let result = decoder.feed_token(9999);
        assert!(matches!(result, Err(DecodeError::UnmappedToken(9999))));
    }

    #[test]
    fn test_invalid_token_error() {
        let mut decoder = make_decoder();
        // Push InputDecl
        decoder.feed_token(5).unwrap();
        // Try to push NodeEnd (invalid after InputDecl)
        let result = decoder.feed_token(4);
        assert!(matches!(result, Err(DecodeError::BuildError(_))));
    }

    #[test]
    fn test_mask_constraints() {
        let mut decoder = make_decoder();

        // Initially, InputDecl, NodeStart, OutputDecl, and Id tokens are valid
        let mask = decoder.next_mask();
        assert!(mask.is_allowed(5)); // InputDecl
        assert!(mask.is_allowed(3)); // NodeStart
        assert!(mask.is_allowed(6)); // OutputDecl

        // After InputDecl, Id tokens should be valid
        decoder.feed_token(5).unwrap(); // InputDecl
        decoder.feed_token(9).unwrap(); // Id(0)

        // After NodeStart Id, only operators are valid
        decoder.feed_token(3).unwrap(); // NodeStart
        decoder.feed_token(10).unwrap(); // Id(1)

        let mask = decoder.next_mask();
        assert_eq!(mask.allowed_count(), 3);
        assert!(mask.is_allowed(0)); // Or
        assert!(mask.is_allowed(1)); // Nor
        assert!(mask.is_allowed(2)); // Xor
        assert!(!mask.is_allowed(3)); // NodeStart - not valid
        assert!(!mask.is_allowed(4)); // NodeEnd - not valid yet
    }

    #[test]
    fn test_full_decode_loop() {
        let mapping = TokenMapping::sequential(256);
        let generator = MaskGenerator::new(mapping.clone(), 1000);
        let mut decoder = ConstrainedDecoder::new(generator);

        // Simulate: "Admin OR (Owner XOR Public)"
        // Tokens: InputDecl Id(0) InputDecl Id(1) InputDecl Id(2)
        //         NodeStart Id(3) Xor Id(1) Id(2) NodeEnd
        //         NodeStart Id(4) Or Id(0) Id(3) NodeEnd
        //         OutputDecl Id(4)
        let tokens = [
            Token::InputDecl,
            Token::Id(0),
            Token::InputDecl,
            Token::Id(1),
            Token::InputDecl,
            Token::Id(2),
            Token::NodeStart,
            Token::Id(3),
            Token::Xor,
            Token::Id(1),
            Token::Id(2),
            Token::NodeEnd,
            Token::NodeStart,
            Token::Id(4),
            Token::Or,
            Token::Id(0),
            Token::Id(3),
            Token::NodeEnd,
            Token::OutputDecl,
            Token::Id(4),
        ];

        for token in tokens {
            let mask = decoder.next_mask();
            let id = mapping.get(token).unwrap();
            assert!(
                mask.is_allowed(id),
                "token {:?} (id {}) should be allowed",
                token,
                id
            );
            decoder.feed_token(id).unwrap();
        }

        let graph = decoder.finish().unwrap();
        assert_eq!(graph.inputs, vec![0, 1, 2]);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.outputs, vec![4]);
    }
}
