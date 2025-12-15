"""Stage 1: Template-based training data expansion."""

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import List, Dict, Tuple
import json


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
    """Get Ministral ID for Id(n)."""
    return 45 + n


# Variable names for template expansion
VARIABLES = [
    "admin", "owner", "public", "authenticated", "verified",
    "member", "guest", "readonly", "premium", "active",
    "enabled", "visible", "approved", "trusted", "internal"
]

# Domain prefixes for diversity
DOMAINS = [
    "Allow access if",
    "Grant permission when",
    "Permit action if",
    "Enable feature when",
    "Show content if",
    "Execute if",
    "Proceed when",
    "Authorize if",
]


@dataclass
class Template:
    """A template for generating training examples."""
    pattern: str  # e.g., "Allow if {a} OR {b}"
    tokens: List[int]  # Ministral token IDs
    var_count: int  # Number of variables needed


def make_tokens(*args) -> List[int]:
    """Flatten token arguments into a list."""
    result = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            result.extend(arg)
        else:
            result.append(arg)
    return result


# Base templates with token sequences
TEMPLATES = [
    # Single input passthrough
    Template(
        pattern="{domain} {a}",
        tokens=make_tokens(INPUT_DECL, Id(0), OUTPUT_DECL, Id(0)),
        var_count=1
    ),

    # NOT: NOT(a) = NOR(a, a)
    Template(
        pattern="{domain} NOT {a}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            NODE_START, Id(1), NOR, Id(0), Id(0), NODE_END,
            OUTPUT_DECL, Id(1)
        ),
        var_count=1
    ),

    # OR: a OR b
    Template(
        pattern="{domain} {a} OR {b}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            NODE_START, Id(2), OR, Id(0), Id(1), NODE_END,
            OUTPUT_DECL, Id(2)
        ),
        var_count=2
    ),

    # XOR: a XOR b
    Template(
        pattern="{domain} {a} XOR {b}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            NODE_START, Id(2), XOR, Id(0), Id(1), NODE_END,
            OUTPUT_DECL, Id(2)
        ),
        var_count=2
    ),

    # NOR: a NOR b (neither)
    Template(
        pattern="{domain} neither {a} nor {b}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            NODE_START, Id(2), NOR, Id(0), Id(1), NODE_END,
            OUTPUT_DECL, Id(2)
        ),
        var_count=2
    ),

    # AND: AND(a, b) = NOR(NOR(a,a), NOR(b,b))
    Template(
        pattern="{domain} {a} AND {b}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            NODE_START, Id(2), NOR, Id(0), Id(0), NODE_END,  # NOT(a)
            NODE_START, Id(3), NOR, Id(1), Id(1), NODE_END,  # NOT(b)
            NODE_START, Id(4), NOR, Id(2), Id(3), NODE_END,  # AND(a, b)
            OUTPUT_DECL, Id(4)
        ),
        var_count=2
    ),

    # IMPLIES: a IMPLIES b = OR(NOT(a), b) = OR(NOR(a,a), b)
    Template(
        pattern="{domain} {a} implies {b}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            NODE_START, Id(2), NOR, Id(0), Id(0), NODE_END,  # NOT(a)
            NODE_START, Id(3), OR, Id(2), Id(1), NODE_END,   # OR(NOT(a), b)
            OUTPUT_DECL, Id(3)
        ),
        var_count=2
    ),

    # Three-input OR: a OR b OR c
    Template(
        pattern="{domain} {a} OR {b} OR {c}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            INPUT_DECL, Id(2),
            NODE_START, Id(3), OR, Id(0), Id(1), NODE_END,  # a OR b
            NODE_START, Id(4), OR, Id(3), Id(2), NODE_END,  # (a OR b) OR c
            OUTPUT_DECL, Id(4)
        ),
        var_count=3
    ),

    # Three-input AND: a AND b AND c
    Template(
        pattern="{domain} {a} AND {b} AND {c}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            INPUT_DECL, Id(2),
            # NOT(a), NOT(b), NOT(c)
            NODE_START, Id(3), NOR, Id(0), Id(0), NODE_END,
            NODE_START, Id(4), NOR, Id(1), Id(1), NODE_END,
            NODE_START, Id(5), NOR, Id(2), Id(2), NODE_END,
            # AND(a, b) = NOR(NOT(a), NOT(b))
            NODE_START, Id(6), NOR, Id(3), Id(4), NODE_END,
            # NOT(AND(a,b))
            NODE_START, Id(7), NOR, Id(6), Id(6), NODE_END,
            # AND(AND(a,b), c) = NOR(NOT(AND(a,b)), NOT(c))
            NODE_START, Id(8), NOR, Id(7), Id(5), NODE_END,
            OUTPUT_DECL, Id(8)
        ),
        var_count=3
    ),

    # Canonical example: a OR (b XOR c)
    Template(
        pattern="{domain} {a} OR ({b} XOR {c})",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            INPUT_DECL, Id(2),
            NODE_START, Id(3), XOR, Id(1), Id(2), NODE_END,  # b XOR c
            NODE_START, Id(4), OR, Id(0), Id(3), NODE_END,   # a OR (b XOR c)
            OUTPUT_DECL, Id(4)
        ),
        var_count=3
    ),

    # (a AND b) OR c
    Template(
        pattern="{domain} ({a} AND {b}) OR {c}",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            INPUT_DECL, Id(2),
            # AND(a, b)
            NODE_START, Id(3), NOR, Id(0), Id(0), NODE_END,  # NOT(a)
            NODE_START, Id(4), NOR, Id(1), Id(1), NODE_END,  # NOT(b)
            NODE_START, Id(5), NOR, Id(3), Id(4), NODE_END,  # AND(a, b)
            # OR with c
            NODE_START, Id(6), OR, Id(5), Id(2), NODE_END,
            OUTPUT_DECL, Id(6)
        ),
        var_count=3
    ),

    # Constant true
    Template(
        pattern="{domain} always true",
        tokens=make_tokens(
            NODE_START, Id(0), OR, TRUE, FALSE, NODE_END,  # true OR false = true
            OUTPUT_DECL, Id(0)
        ),
        var_count=0
    ),

    # Constant false
    Template(
        pattern="{domain} always false",
        tokens=make_tokens(
            NODE_START, Id(0), NOR, TRUE, TRUE, NODE_END,  # true NOR true = false
            OUTPUT_DECL, Id(0)
        ),
        var_count=0
    ),

    # a OR constant
    Template(
        pattern="{domain} {a} OR true",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            NODE_START, Id(1), OR, Id(0), TRUE, NODE_END,
            OUTPUT_DECL, Id(1)
        ),
        var_count=1
    ),

    Template(
        pattern="{domain} {a} AND false",
        tokens=make_tokens(
            INPUT_DECL, Id(0),
            # AND(a, false) = NOR(NOT(a), NOT(false)) = NOR(NOT(a), true)
            NODE_START, Id(1), NOR, Id(0), Id(0), NODE_END,  # NOT(a)
            NODE_START, Id(2), NOR, Id(1), TRUE, NODE_END,   # AND(a, false)
            OUTPUT_DECL, Id(2)
        ),
        var_count=1
    ),
]


def expand_template(template: Template, variables: List[str], domain: str) -> Dict:
    """Expand a template with specific variables and domain."""
    var_map = {f"{{{'abcdefghij'[i]}}}": var for i, var in enumerate(variables)}
    var_map["{domain}"] = domain

    prompt = template.pattern
    for key, value in var_map.items():
        prompt = prompt.replace(key, value)

    return {
        "prompt": prompt,
        "completion": " ".join(str(t) for t in template.tokens)
    }


def generate_template_examples(max_per_template: int = 100) -> List[Dict]:
    """Generate training examples from templates."""
    examples = []

    for template in TEMPLATES:
        count = 0

        if template.var_count == 0:
            # Constant templates - one per domain
            for domain in DOMAINS:
                examples.append(expand_template(template, [], domain))
                count += 1
                if count >= max_per_template:
                    break

        elif template.var_count == 1:
            # Single variable - all variables, all domains
            for domain in DOMAINS:
                for var in VARIABLES:
                    if count >= max_per_template:
                        break
                    examples.append(expand_template(template, [var], domain))
                    count += 1

        elif template.var_count == 2:
            # Two variables - combinations
            for domain in DOMAINS[:4]:  # Limit domains
                for var_combo in combinations(VARIABLES[:10], 2):
                    if count >= max_per_template:
                        break
                    examples.append(expand_template(template, list(var_combo), domain))
                    count += 1

        elif template.var_count == 3:
            # Three variables - limited combinations
            for domain in DOMAINS[:2]:
                for var_combo in combinations(VARIABLES[:8], 3):
                    if count >= max_per_template:
                        break
                    examples.append(expand_template(template, list(var_combo), domain))
                    count += 1

    return examples


def main():
    """Generate template-expanded examples and write to JSONL."""
    examples = generate_template_examples(max_per_template=80)

    print(f"Generated {len(examples)} template examples")

    # Write to file
    with open("template_examples.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    # Print a few samples
    print("\nSample examples:")
    for example in examples[:5]:
        print(f"  {example['prompt']}")
        print(f"    -> {example['completion'][:60]}...")


if __name__ == "__main__":
    main()
