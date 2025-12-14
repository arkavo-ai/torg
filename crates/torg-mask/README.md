# torg-mask

LLM logit masking for TØR-G constrained decoding.

[![Crates.io](https://img.shields.io/crates/v/torg-mask.svg)](https://crates.io/crates/torg-mask)
[![Documentation](https://docs.rs/torg-mask/badge.svg)](https://docs.rs/torg-mask)

## Overview

This crate provides utilities for integrating TØR-G's `valid_next_tokens()` with LLM inference engines. By masking invalid tokens during decoding, LLMs are guaranteed to produce syntactically correct boolean circuits.

## Features

- **Token Mapping**: Bidirectional mapping between TØR-G tokens and LLM vocab IDs
- **Logit Masking**: Efficient O(vocab_size + allowed_count) mask application
- **Constrained Decoder**: High-level orchestrator for decode loops

## Usage

```rust
use torg_mask::{ConstrainedDecoder, MaskGenerator, TokenMapping};

// Create a token mapping
// In production, map to actual unused token IDs in your LLM's vocabulary
let mapping = TokenMapping::sequential(256);
let generator = MaskGenerator::new(mapping.clone(), vocab_size);
let mut decoder = ConstrainedDecoder::new(generator);

// Decode loop
while !decoder.is_complete() {
    // Get mask for valid tokens
    let mask = decoder.next_mask();

    // Apply mask to LLM logits
    mask.apply_to_logits(&mut logits);

    // Sample from masked distribution
    let token_id = sample(&logits);

    // Feed token to decoder
    decoder.feed_token(token_id)?;
}

// Get the constructed graph
let graph = decoder.finish()?;
```

## Token Mapping

The `TokenMapping` struct maps TØR-G's 9 fixed tokens plus Id tokens to LLM vocabulary IDs:

```rust
// Sequential mapping for testing (IDs 0-264)
let mapping = TokenMapping::sequential(256);

// Custom mapping for production
let mapping = TokenMapping::builder()
    .or(50256)           // Map Or to vocab ID 50256
    .nor(50257)
    .xor(50258)
    .node_start(50259)
    .node_end(50260)
    .input_decl(50261)
    .output_decl(50262)
    .true_token(50263)
    .false_token(50264)
    .id_base(50265)      // Id(n) maps to 50265 + n
    .id_count(256)
    .build();
```

## Logit Masking

The `LogitMask` struct efficiently applies masks to logit vectors:

```rust
let mask = generator.generate(&valid_tokens);

// Check if a token is allowed
if mask.is_allowed(token_id) { ... }

// Apply mask in-place (O(vocab_size + allowed_count))
mask.apply_to_logits(&mut logits);

// Get allowed token IDs
for &id in mask.allowed_ids() { ... }
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
