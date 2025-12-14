//! Logit mask for LLM inference.

/// A logit mask specifying which tokens are allowed.
///
/// This struct holds a sorted list of allowed LLM token IDs and provides
/// efficient methods for applying the mask to logit vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitMask {
    vocab_size: usize,
    /// Allowed token IDs, sorted for efficient lookup.
    allowed: Vec<u32>,
}

impl LogitMask {
    /// Create a new logit mask.
    ///
    /// The `allowed` vector will be sorted internally.
    pub fn new(vocab_size: usize, mut allowed: Vec<u32>) -> Self {
        allowed.sort_unstable();
        allowed.dedup();
        Self {
            vocab_size,
            allowed,
        }
    }

    /// Create a mask that allows all tokens.
    pub fn allow_all(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            allowed: (0..vocab_size as u32).collect(),
        }
    }

    /// Create a mask that allows no tokens.
    pub fn allow_none(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            allowed: Vec::new(),
        }
    }

    /// Check if a token ID is allowed.
    pub fn is_allowed(&self, id: u32) -> bool {
        self.allowed.binary_search(&id).is_ok()
    }

    /// Get the allowed token IDs.
    pub fn allowed_ids(&self) -> &[u32] {
        &self.allowed
    }

    /// Get the number of allowed tokens.
    pub fn allowed_count(&self) -> usize {
        self.allowed.len()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Check if no tokens are allowed.
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }

    /// Apply the mask to a logit vector.
    ///
    /// Sets all disallowed token logits to negative infinity.
    /// This is O(vocab_size + allowed_count), not O(vocab_size * log(allowed_count)).
    ///
    /// # Panics
    ///
    /// Panics if `logits.len() != vocab_size`.
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        assert_eq!(
            logits.len(),
            self.vocab_size,
            "logits length {} != vocab_size {}",
            logits.len(),
            self.vocab_size
        );

        if self.allowed.is_empty() {
            logits.fill(f32::NEG_INFINITY);
            return;
        }

        // Save the allowed logits
        let originals: Vec<f32> = self.allowed.iter().map(|&id| logits[id as usize]).collect();

        // Set all to -inf
        logits.fill(f32::NEG_INFINITY);

        // Restore the allowed ones
        for (i, &id) in self.allowed.iter().enumerate() {
            logits[id as usize] = originals[i];
        }
    }

    /// Apply the mask and return a new logit vector.
    ///
    /// This is a convenience method that clones the input.
    pub fn apply_to_logits_cloned(&self, logits: &[f32]) -> Vec<f32> {
        let mut result = logits.to_vec();
        self.apply_to_logits(&mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_allowed() {
        let mask = LogitMask::new(100, vec![5, 10, 15, 20]);

        assert!(mask.is_allowed(5));
        assert!(mask.is_allowed(10));
        assert!(mask.is_allowed(15));
        assert!(mask.is_allowed(20));
        assert!(!mask.is_allowed(0));
        assert!(!mask.is_allowed(6));
        assert!(!mask.is_allowed(99));
    }

    #[test]
    fn test_apply_to_logits() {
        let mask = LogitMask::new(10, vec![2, 5, 7]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        mask.apply_to_logits(&mut logits);

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0); // Allowed
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
        assert_eq!(logits[5], 6.0); // Allowed
        assert_eq!(logits[6], f32::NEG_INFINITY);
        assert_eq!(logits[7], 8.0); // Allowed
        assert_eq!(logits[8], f32::NEG_INFINITY);
        assert_eq!(logits[9], f32::NEG_INFINITY);
    }

    #[test]
    fn test_empty_mask() {
        let mask = LogitMask::allow_none(10);
        let mut logits = vec![1.0; 10];

        mask.apply_to_logits(&mut logits);

        for &l in &logits {
            assert_eq!(l, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_single_allowed() {
        let mask = LogitMask::new(10, vec![3]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        mask.apply_to_logits(&mut logits);

        assert_eq!(logits[3], 4.0);
        for (i, &l) in logits.iter().enumerate() {
            if i != 3 {
                assert_eq!(l, f32::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn test_dedup() {
        let mask = LogitMask::new(10, vec![5, 5, 5, 3, 3]);
        assert_eq!(mask.allowed_ids(), &[3, 5]);
    }

    #[test]
    #[should_panic(expected = "logits length")]
    fn test_wrong_size_panics() {
        let mask = LogitMask::new(10, vec![5]);
        let mut logits = vec![1.0; 5]; // Wrong size
        mask.apply_to_logits(&mut logits);
    }
}
