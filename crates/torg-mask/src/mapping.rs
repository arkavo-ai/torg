//! Token mapping between TØR-G tokens and LLM vocabulary IDs.

use torg_core::Token;

/// Maps TØR-G tokens to LLM vocabulary token IDs.
///
/// TØR-G has 9 fixed tokens plus a range of Id tokens. This struct
/// provides bidirectional mapping between TØR-G tokens and the
/// corresponding token IDs in an LLM's vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMapping {
    // Fixed token mappings
    or_id: u32,
    nor_id: u32,
    xor_id: u32,
    node_start_id: u32,
    node_end_id: u32,
    input_decl_id: u32,
    output_decl_id: u32,
    true_id: u32,
    false_id: u32,

    /// Base LLM token ID for Id(0). Id(n) maps to id_base + n.
    id_base: u32,

    /// Number of Id tokens mapped (0..id_count).
    /// Default 256, matching typical Limits. This is bounded by
    /// practical usage (Limits::max_inputs + max_nodes), not the u16 type range.
    id_count: u16,
}

impl TokenMapping {
    /// Create a builder for custom token mappings.
    pub fn builder() -> TokenMappingBuilder {
        TokenMappingBuilder::new()
    }

    /// Example mapping using sequential IDs starting from 0.
    ///
    /// **Warning**: This is for testing only. In production, you must
    /// map to actual unused token IDs in your LLM's vocabulary.
    ///
    /// Layout:
    /// - 0: Or
    /// - 1: Nor
    /// - 2: Xor
    /// - 3: NodeStart
    /// - 4: NodeEnd
    /// - 5: InputDecl
    /// - 6: OutputDecl
    /// - 7: True
    /// - 8: False
    /// - 9..265: Id(0)..Id(255)
    pub fn sequential(id_count: u16) -> Self {
        Self {
            or_id: 0,
            nor_id: 1,
            xor_id: 2,
            node_start_id: 3,
            node_end_id: 4,
            input_decl_id: 5,
            output_decl_id: 6,
            true_id: 7,
            false_id: 8,
            id_base: 9,
            id_count,
        }
    }

    /// Mapping for Mistral/Ministral models using reserved `<SPECIAL_N>` tokens.
    ///
    /// Ministral tokenizers reserve token IDs 36-565 as `<SPECIAL_N>` placeholders.
    /// This mapping uses IDs 36-300 for TØR-G tokens:
    ///
    /// | TØR-G Token | Ministral ID |
    /// |-------------|--------------|
    /// | `Or`        | 36           |
    /// | `Nor`       | 37           |
    /// | `Xor`       | 38           |
    /// | `NodeStart` | 39           |
    /// | `NodeEnd`   | 40           |
    /// | `InputDecl` | 41           |
    /// | `OutputDecl`| 42           |
    /// | `True`      | 43           |
    /// | `False`     | 44           |
    /// | `Id(0)`     | 45           |
    /// | `Id(255)`   | 300          |
    ///
    /// Compatible with: Ministral-3B, Ministral-8B, Mistral-7B v0.3+
    pub fn ministral() -> Self {
        Self {
            or_id: 36,
            nor_id: 37,
            xor_id: 38,
            node_start_id: 39,
            node_end_id: 40,
            input_decl_id: 41,
            output_decl_id: 42,
            true_id: 43,
            false_id: 44,
            id_base: 45,
            id_count: 256,
        }
    }

    /// Map a TØR-G token to its LLM vocabulary ID.
    ///
    /// Returns `None` if the token cannot be mapped (e.g., Id out of range).
    pub fn get(&self, token: Token) -> Option<u32> {
        match token {
            Token::Or => Some(self.or_id),
            Token::Nor => Some(self.nor_id),
            Token::Xor => Some(self.xor_id),
            Token::NodeStart => Some(self.node_start_id),
            Token::NodeEnd => Some(self.node_end_id),
            Token::InputDecl => Some(self.input_decl_id),
            Token::OutputDecl => Some(self.output_decl_id),
            Token::True => Some(self.true_id),
            Token::False => Some(self.false_id),
            Token::Id(n) => {
                if n < self.id_count {
                    Some(self.id_base + n as u32)
                } else {
                    None
                }
            }
        }
    }

    /// Map an LLM vocabulary ID back to a TØR-G token.
    ///
    /// Returns `None` if the ID doesn't correspond to any mapped token.
    pub fn reverse(&self, id: u32) -> Option<Token> {
        if id == self.or_id {
            Some(Token::Or)
        } else if id == self.nor_id {
            Some(Token::Nor)
        } else if id == self.xor_id {
            Some(Token::Xor)
        } else if id == self.node_start_id {
            Some(Token::NodeStart)
        } else if id == self.node_end_id {
            Some(Token::NodeEnd)
        } else if id == self.input_decl_id {
            Some(Token::InputDecl)
        } else if id == self.output_decl_id {
            Some(Token::OutputDecl)
        } else if id == self.true_id {
            Some(Token::True)
        } else if id == self.false_id {
            Some(Token::False)
        } else if id >= self.id_base && id < self.id_base + self.id_count as u32 {
            Some(Token::Id((id - self.id_base) as u16))
        } else {
            None
        }
    }

    /// Get the number of Id tokens mapped.
    pub fn id_count(&self) -> u16 {
        self.id_count
    }

    /// Get the total number of mapped tokens (9 fixed + id_count).
    pub fn total_tokens(&self) -> usize {
        9 + self.id_count as usize
    }
}

impl Default for TokenMapping {
    fn default() -> Self {
        Self::sequential(256)
    }
}

/// Builder for creating custom token mappings.
#[derive(Debug, Clone)]
pub struct TokenMappingBuilder {
    or_id: Option<u32>,
    nor_id: Option<u32>,
    xor_id: Option<u32>,
    node_start_id: Option<u32>,
    node_end_id: Option<u32>,
    input_decl_id: Option<u32>,
    output_decl_id: Option<u32>,
    true_id: Option<u32>,
    false_id: Option<u32>,
    id_base: Option<u32>,
    id_count: u16,
}

impl TokenMappingBuilder {
    /// Create a new builder with no mappings set.
    pub fn new() -> Self {
        Self {
            or_id: None,
            nor_id: None,
            xor_id: None,
            node_start_id: None,
            node_end_id: None,
            input_decl_id: None,
            output_decl_id: None,
            true_id: None,
            false_id: None,
            id_base: None,
            id_count: 256,
        }
    }

    /// Set the LLM token ID for Or.
    pub fn or(mut self, id: u32) -> Self {
        self.or_id = Some(id);
        self
    }

    /// Set the LLM token ID for Nor.
    pub fn nor(mut self, id: u32) -> Self {
        self.nor_id = Some(id);
        self
    }

    /// Set the LLM token ID for Xor.
    pub fn xor(mut self, id: u32) -> Self {
        self.xor_id = Some(id);
        self
    }

    /// Set the LLM token ID for NodeStart.
    pub fn node_start(mut self, id: u32) -> Self {
        self.node_start_id = Some(id);
        self
    }

    /// Set the LLM token ID for NodeEnd.
    pub fn node_end(mut self, id: u32) -> Self {
        self.node_end_id = Some(id);
        self
    }

    /// Set the LLM token ID for InputDecl.
    pub fn input_decl(mut self, id: u32) -> Self {
        self.input_decl_id = Some(id);
        self
    }

    /// Set the LLM token ID for OutputDecl.
    pub fn output_decl(mut self, id: u32) -> Self {
        self.output_decl_id = Some(id);
        self
    }

    /// Set the LLM token ID for True.
    pub fn true_token(mut self, id: u32) -> Self {
        self.true_id = Some(id);
        self
    }

    /// Set the LLM token ID for False.
    pub fn false_token(mut self, id: u32) -> Self {
        self.false_id = Some(id);
        self
    }

    /// Set the base LLM token ID for Id tokens.
    /// Id(n) will map to id_base + n.
    pub fn id_base(mut self, base: u32) -> Self {
        self.id_base = Some(base);
        self
    }

    /// Set how many Id tokens to map (default 256).
    pub fn id_count(mut self, count: u16) -> Self {
        self.id_count = count;
        self
    }

    /// Build the token mapping.
    ///
    /// # Panics
    ///
    /// Panics if any required mapping is not set.
    pub fn build(self) -> TokenMapping {
        TokenMapping {
            or_id: self.or_id.expect("or_id not set"),
            nor_id: self.nor_id.expect("nor_id not set"),
            xor_id: self.xor_id.expect("xor_id not set"),
            node_start_id: self.node_start_id.expect("node_start_id not set"),
            node_end_id: self.node_end_id.expect("node_end_id not set"),
            input_decl_id: self.input_decl_id.expect("input_decl_id not set"),
            output_decl_id: self.output_decl_id.expect("output_decl_id not set"),
            true_id: self.true_id.expect("true_id not set"),
            false_id: self.false_id.expect("false_id not set"),
            id_base: self.id_base.expect("id_base not set"),
            id_count: self.id_count,
        }
    }
}

impl Default for TokenMappingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_mapping() {
        let mapping = TokenMapping::sequential(256);

        assert_eq!(mapping.get(Token::Or), Some(0));
        assert_eq!(mapping.get(Token::Nor), Some(1));
        assert_eq!(mapping.get(Token::Xor), Some(2));
        assert_eq!(mapping.get(Token::NodeStart), Some(3));
        assert_eq!(mapping.get(Token::NodeEnd), Some(4));
        assert_eq!(mapping.get(Token::InputDecl), Some(5));
        assert_eq!(mapping.get(Token::OutputDecl), Some(6));
        assert_eq!(mapping.get(Token::True), Some(7));
        assert_eq!(mapping.get(Token::False), Some(8));
        assert_eq!(mapping.get(Token::Id(0)), Some(9));
        assert_eq!(mapping.get(Token::Id(255)), Some(264));
        assert_eq!(mapping.get(Token::Id(256)), None);
    }

    #[test]
    fn test_reverse_mapping() {
        let mapping = TokenMapping::sequential(256);

        assert_eq!(mapping.reverse(0), Some(Token::Or));
        assert_eq!(mapping.reverse(1), Some(Token::Nor));
        assert_eq!(mapping.reverse(9), Some(Token::Id(0)));
        assert_eq!(mapping.reverse(264), Some(Token::Id(255)));
        assert_eq!(mapping.reverse(265), None);
        assert_eq!(mapping.reverse(1000), None);
    }

    #[test]
    fn test_round_trip() {
        let mapping = TokenMapping::sequential(256);

        let tokens = [
            Token::Or,
            Token::Nor,
            Token::Xor,
            Token::NodeStart,
            Token::NodeEnd,
            Token::InputDecl,
            Token::OutputDecl,
            Token::True,
            Token::False,
            Token::Id(0),
            Token::Id(42),
            Token::Id(255),
        ];

        for token in tokens {
            let id = mapping.get(token).unwrap();
            let back = mapping.reverse(id).unwrap();
            assert_eq!(token, back);
        }
    }

    #[test]
    fn test_builder() {
        let mapping = TokenMapping::builder()
            .or(100)
            .nor(101)
            .xor(102)
            .node_start(103)
            .node_end(104)
            .input_decl(105)
            .output_decl(106)
            .true_token(107)
            .false_token(108)
            .id_base(1000)
            .id_count(128)
            .build();

        assert_eq!(mapping.get(Token::Or), Some(100));
        assert_eq!(mapping.get(Token::Id(0)), Some(1000));
        assert_eq!(mapping.get(Token::Id(127)), Some(1127));
        assert_eq!(mapping.get(Token::Id(128)), None);
    }

    #[test]
    fn test_ministral_mapping() {
        let mapping = TokenMapping::ministral();

        // Fixed tokens use reserved <SPECIAL_N> IDs 36-44
        assert_eq!(mapping.get(Token::Or), Some(36));
        assert_eq!(mapping.get(Token::Nor), Some(37));
        assert_eq!(mapping.get(Token::Xor), Some(38));
        assert_eq!(mapping.get(Token::NodeStart), Some(39));
        assert_eq!(mapping.get(Token::NodeEnd), Some(40));
        assert_eq!(mapping.get(Token::InputDecl), Some(41));
        assert_eq!(mapping.get(Token::OutputDecl), Some(42));
        assert_eq!(mapping.get(Token::True), Some(43));
        assert_eq!(mapping.get(Token::False), Some(44));

        // Id tokens start at 45
        assert_eq!(mapping.get(Token::Id(0)), Some(45));
        assert_eq!(mapping.get(Token::Id(255)), Some(300));
        assert_eq!(mapping.get(Token::Id(256)), None);

        // Total: 9 fixed + 256 Id = 265 tokens
        assert_eq!(mapping.total_tokens(), 265);
    }

    #[test]
    fn test_ministral_round_trip() {
        let mapping = TokenMapping::ministral();

        // Test all fixed tokens
        for token in [
            Token::Or,
            Token::Nor,
            Token::Xor,
            Token::NodeStart,
            Token::NodeEnd,
            Token::InputDecl,
            Token::OutputDecl,
            Token::True,
            Token::False,
        ] {
            let id = mapping.get(token).unwrap();
            let back = mapping.reverse(id).unwrap();
            assert_eq!(token, back);
        }

        // Test Id tokens at boundaries
        for n in [0, 1, 127, 128, 254, 255] {
            let token = Token::Id(n);
            let id = mapping.get(token).unwrap();
            let back = mapping.reverse(id).unwrap();
            assert_eq!(token, back);
        }
    }
}
