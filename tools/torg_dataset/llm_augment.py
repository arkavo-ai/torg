#!/usr/bin/env python3
"""Stage 3: LLM-assisted training data generation.

Uses Claude API to generate diverse policy descriptions and their
corresponding TØR-G token sequences.
"""

import json
import os
import sys
from typing import List, Dict, Optional

try:
    import anthropic
except ImportError:
    anthropic = None


# Ministral token IDs
OR = 36
NOR = 37
XOR = 38
NODE_START = 39
NODE_END = 40
INPUT_DECL = 41
OUTPUT_DECL = 42
TRUE = 43
FALSE = 44

def Id(n: int) -> int:
    return 45 + n


SYSTEM_PROMPT = """You are a TØR-G (Token-Only Reasoner - Graph) expert. TØR-G is a boolean-circuit IR that LLMs emit as constrained token sequences.

## Token Vocabulary

| Token | Ministral ID | Purpose |
|-------|--------------|---------|
| Or | 36 | Binary OR operator |
| Nor | 37 | Binary NOR operator |
| Xor | 38 | Binary XOR operator |
| NodeStart | 39 | Begin node definition |
| NodeEnd | 40 | End node definition |
| InputDecl | 41 | Declare input variable |
| OutputDecl | 42 | Declare output |
| True | 43 | Constant true |
| False | 44 | Constant false |
| Id(n) | 45+n | Reference to input/node n |

## Grammar

```
graph   = inputs nodes outputs
inputs  = (InputDecl Id)*
nodes   = (NodeStart Id Op Source Source NodeEnd)*
outputs = (OutputDecl Id)+
Op      = Or | Nor | Xor
Source  = Id | True | False
```

## Derived Operations

- NOT(a) = NOR(a, a)
- AND(a, b) = NOR(NOR(a,a), NOR(b,b))
- IMPLIES(a, b) = OR(NOR(a,a), b)

## Key Rules

1. Each node has exactly 2 operands (binary operators only)
2. DAG structure: nodes can only reference previously-defined IDs
3. Id(n) refers to the n-th declared input or node
4. Inputs are declared first, then nodes, then outputs

## Example

Policy: "Allow if admin OR (owner XOR public)"

Token sequence:
41 45 41 46 41 47 39 48 38 46 47 40 39 49 36 45 48 40 42 49

Explanation:
- 41 45: InputDecl Id(0) - declare "admin" as input 0
- 41 46: InputDecl Id(1) - declare "owner" as input 1
- 41 47: InputDecl Id(2) - declare "public" as input 2
- 39 48 38 46 47 40: NodeStart Id(3) Xor Id(1) Id(2) NodeEnd - owner XOR public
- 39 49 36 45 48 40: NodeStart Id(4) Or Id(0) Id(3) NodeEnd - admin OR node(3)
- 42 49: OutputDecl Id(4) - output the result
"""


def generate_diverse_policies(
    client: "anthropic.Anthropic",
    base_examples: List[Dict],
    count: int = 50,
    domain: str = "access control",
) -> List[Dict]:
    """Generate diverse policy examples using Claude."""

    # Format base examples for the prompt
    examples_text = "\n".join([
        f"- Prompt: \"{ex['prompt']}\"\n  Tokens: {ex['completion']}"
        for ex in base_examples[:5]
    ])

    user_prompt = f"""Generate {count} new, diverse TØR-G policy examples for the domain: {domain}

Here are some reference examples:
{examples_text}

For each new example:
1. Write a natural language policy description
2. Generate the corresponding Ministral token ID sequence

Requirements:
- Vary complexity: some simple (2-3 inputs), some complex (4-6 inputs)
- Use diverse variable names relevant to {domain}
- Include edge cases: constants only, multiple outputs, deep chains
- Ensure DAG property: nodes only reference earlier-defined IDs

Output format (JSON array):
[
  {{"prompt": "policy description", "completion": "space-separated token IDs"}},
  ...
]

Generate exactly {count} examples as a JSON array:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    # Parse the response
    content = response.content[0].text

    # Extract JSON array from response
    try:
        # Find JSON array in response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            print(f"Warning: Could not find JSON array in response")
            return []

        json_str = content[start:end]
        examples = json.loads(json_str)
        return examples
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return []


def validate_token_sequence(completion: str) -> bool:
    """Basic validation of a token sequence."""
    try:
        tokens = [int(t) for t in completion.split()]
    except ValueError:
        return False

    if not tokens:
        return False

    # Must have OUTPUT_DECL (42)
    if OUTPUT_DECL not in tokens:
        return False

    # Must end with an Id token
    if tokens[-1] < 45:
        return False

    # Check for valid token IDs
    for t in tokens:
        if t < 36 or t > 300:  # 36-44 fixed, 45-300 Id tokens
            return False

    return True


def generate_augmented_examples(
    api_key: Optional[str] = None,
    count: int = 500,
    domains: Optional[List[str]] = None,
) -> List[Dict]:
    """Generate LLM-augmented training examples."""

    if anthropic is None:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return []

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    domains = domains or [
        "access control",
        "feature flags",
        "workflow automation",
        "content moderation",
        "resource permissions",
        "approval workflows",
    ]

    # Load some base examples
    from golden import get_golden_examples
    base_examples = get_golden_examples()

    all_examples = []
    per_domain = count // len(domains)

    for domain in domains:
        print(f"Generating {per_domain} examples for domain: {domain}")
        try:
            examples = generate_diverse_policies(
                client, base_examples, count=per_domain, domain=domain
            )

            # Validate
            valid = []
            for ex in examples:
                if validate_token_sequence(ex.get("completion", "")):
                    valid.append(ex)
                else:
                    print(f"  Skipped invalid: {ex.get('prompt', '')[:50]}...")

            all_examples.extend(valid)
            print(f"  Generated {len(valid)} valid examples")

        except Exception as e:
            print(f"  Error: {e}")

    return all_examples


def main():
    """Generate LLM-augmented examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate LLM-augmented TØR-G examples")
    parser.add_argument("--count", type=int, default=100, help="Total examples to generate")
    parser.add_argument("--output", type=str, default="llm_augmented.jsonl", help="Output file")
    parser.add_argument("--api-key", type=str, help="Anthropic API key (or set ANTHROPIC_API_KEY)")

    args = parser.parse_args()

    examples = generate_augmented_examples(
        api_key=args.api_key,
        count=args.count,
    )

    if examples:
        with open(args.output, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"\nWrote {len(examples)} examples to {args.output}")
    else:
        print("No examples generated")


if __name__ == "__main__":
    main()
