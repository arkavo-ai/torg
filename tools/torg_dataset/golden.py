"""Stage 2: Golden examples from the TØR-G spec and tests."""

from typing import List, Dict
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
    return 45 + n


# Golden examples from spec and golden_vectors.rs
GOLDEN_EXAMPLES = [
    # Canonical example from README
    {
        "prompt": "Allow if admin OR (owner XOR public)",
        "description": "Admin overrides, owner XOR public for others",
        "tokens": [
            INPUT_DECL, Id(0),  # is_admin
            INPUT_DECL, Id(1),  # is_owner
            INPUT_DECL, Id(2),  # is_public
            NODE_START, Id(3), XOR, Id(1), Id(2), NODE_END,  # owner XOR public
            NODE_START, Id(4), OR, Id(0), Id(3), NODE_END,   # admin OR node(3)
            OUTPUT_DECL, Id(4),
        ]
    },

    # AND via NOR gates
    {
        "prompt": "Allow if user is authenticated AND user is verified",
        "description": "AND(a, b) = NOR(NOR(a,a), NOR(b,b))",
        "tokens": [
            INPUT_DECL, Id(0),  # authenticated
            INPUT_DECL, Id(1),  # verified
            NODE_START, Id(2), NOR, Id(0), Id(0), NODE_END,  # NOT(a)
            NODE_START, Id(3), NOR, Id(1), Id(1), NODE_END,  # NOT(b)
            NODE_START, Id(4), NOR, Id(2), Id(3), NODE_END,  # AND(a, b)
            OUTPUT_DECL, Id(4),
        ]
    },

    # NOT via NOR
    {
        "prompt": "Deny if user is banned",
        "description": "NOT(a) = NOR(a, a)",
        "tokens": [
            INPUT_DECL, Id(0),  # is_banned
            NODE_START, Id(1), NOR, Id(0), Id(0), NODE_END,
            OUTPUT_DECL, Id(1),
        ]
    },

    # Constants only
    {
        "prompt": "Always allow access",
        "description": "Constant true via XOR",
        "tokens": [
            NODE_START, Id(0), XOR, TRUE, FALSE, NODE_END,
            OUTPUT_DECL, Id(0),
        ]
    },

    # Multiple outputs
    {
        "prompt": "Output read permission (admin OR member) and write permission (admin AND owner)",
        "description": "Two outputs from shared inputs",
        "tokens": [
            INPUT_DECL, Id(0),  # is_admin
            INPUT_DECL, Id(1),  # is_member
            INPUT_DECL, Id(2),  # is_owner
            # Read: admin OR member
            NODE_START, Id(3), OR, Id(0), Id(1), NODE_END,
            # Write: admin AND owner
            NODE_START, Id(4), NOR, Id(0), Id(0), NODE_END,  # NOT(admin)
            NODE_START, Id(5), NOR, Id(2), Id(2), NODE_END,  # NOT(owner)
            NODE_START, Id(6), NOR, Id(4), Id(5), NODE_END,  # AND(admin, owner)
            OUTPUT_DECL, Id(3),  # read permission
            OUTPUT_DECL, Id(6),  # write permission
        ]
    },

    # Input as output (passthrough)
    {
        "prompt": "Allow if enabled flag is set",
        "description": "Direct input passthrough",
        "tokens": [
            INPUT_DECL, Id(0),
            OUTPUT_DECL, Id(0),
        ]
    },

    # Deep chain
    {
        "prompt": "Allow if any of the four permission levels is granted",
        "description": "Chain of OR operations",
        "tokens": [
            INPUT_DECL, Id(0),  # level1
            INPUT_DECL, Id(1),  # level2
            INPUT_DECL, Id(2),  # level3
            INPUT_DECL, Id(3),  # level4
            NODE_START, Id(4), OR, Id(0), Id(1), NODE_END,
            NODE_START, Id(5), OR, Id(4), Id(2), NODE_END,
            NODE_START, Id(6), OR, Id(5), Id(3), NODE_END,
            OUTPUT_DECL, Id(6),
        ]
    },

    # NAND via NOR
    {
        "prompt": "Deny if both conditions are true (NAND)",
        "description": "NAND(a, b) = NOT(AND(a, b))",
        "tokens": [
            INPUT_DECL, Id(0),
            INPUT_DECL, Id(1),
            # AND(a, b)
            NODE_START, Id(2), NOR, Id(0), Id(0), NODE_END,
            NODE_START, Id(3), NOR, Id(1), Id(1), NODE_END,
            NODE_START, Id(4), NOR, Id(2), Id(3), NODE_END,
            # NOT(AND)
            NODE_START, Id(5), NOR, Id(4), Id(4), NODE_END,
            OUTPUT_DECL, Id(5),
        ]
    },

    # IMPLIES
    {
        "prompt": "If user is member then user must be verified (member implies verified)",
        "description": "IMPLIES(a, b) = OR(NOT(a), b)",
        "tokens": [
            INPUT_DECL, Id(0),  # is_member
            INPUT_DECL, Id(1),  # is_verified
            NODE_START, Id(2), NOR, Id(0), Id(0), NODE_END,  # NOT(member)
            NODE_START, Id(3), OR, Id(2), Id(1), NODE_END,   # OR(NOT(member), verified)
            OUTPUT_DECL, Id(3),
        ]
    },

    # XOR with constant
    {
        "prompt": "Invert the flag value (XOR with true)",
        "description": "NOT via XOR with true",
        "tokens": [
            INPUT_DECL, Id(0),
            NODE_START, Id(1), XOR, Id(0), TRUE, NODE_END,
            OUTPUT_DECL, Id(1),
        ]
    },

    # Complex nested expression
    {
        "prompt": "Allow if (admin OR owner) AND (active OR verified)",
        "description": "Two OR nodes combined with AND",
        "tokens": [
            INPUT_DECL, Id(0),  # admin
            INPUT_DECL, Id(1),  # owner
            INPUT_DECL, Id(2),  # active
            INPUT_DECL, Id(3),  # verified
            # admin OR owner
            NODE_START, Id(4), OR, Id(0), Id(1), NODE_END,
            # active OR verified
            NODE_START, Id(5), OR, Id(2), Id(3), NODE_END,
            # AND the two results
            NODE_START, Id(6), NOR, Id(4), Id(4), NODE_END,  # NOT(admin OR owner)
            NODE_START, Id(7), NOR, Id(5), Id(5), NODE_END,  # NOT(active OR verified)
            NODE_START, Id(8), NOR, Id(6), Id(7), NODE_END,  # AND
            OUTPUT_DECL, Id(8),
        ]
    },

    # Exactly one (XOR for exclusivity)
    {
        "prompt": "Allow if exactly one of admin or guest is true",
        "description": "XOR for mutual exclusivity",
        "tokens": [
            INPUT_DECL, Id(0),  # admin
            INPUT_DECL, Id(1),  # guest
            NODE_START, Id(2), XOR, Id(0), Id(1), NODE_END,
            OUTPUT_DECL, Id(2),
        ]
    },

    # Feature flag pattern
    {
        "prompt": "Enable feature if beta flag is set OR user is premium",
        "description": "Feature flag rollout pattern",
        "tokens": [
            INPUT_DECL, Id(0),  # beta_flag
            INPUT_DECL, Id(1),  # is_premium
            NODE_START, Id(2), OR, Id(0), Id(1), NODE_END,
            OUTPUT_DECL, Id(2),
        ]
    },

    # Simpler version of role hierarchy
    {
        "prompt": "Allow if superadmin OR (admin AND approved)",
        "description": "Role hierarchy pattern",
        "tokens": [
            INPUT_DECL, Id(0),  # superadmin
            INPUT_DECL, Id(1),  # admin
            INPUT_DECL, Id(2),  # approved
            # AND(admin, approved)
            NODE_START, Id(3), NOR, Id(1), Id(1), NODE_END,  # NOT(admin)
            NODE_START, Id(4), NOR, Id(2), Id(2), NODE_END,  # NOT(approved)
            NODE_START, Id(5), NOR, Id(3), Id(4), NODE_END,  # AND(admin, approved)
            # superadmin OR result
            NODE_START, Id(6), OR, Id(0), Id(5), NODE_END,
            OUTPUT_DECL, Id(6),
        ]
    },

    # Workflow gate pattern
    {
        "prompt": "Proceed to next stage if current stage is complete AND no blockers exist",
        "description": "Workflow gate check",
        "tokens": [
            INPUT_DECL, Id(0),  # stage_complete
            INPUT_DECL, Id(1),  # has_blockers
            # NOT(has_blockers)
            NODE_START, Id(2), NOR, Id(1), Id(1), NODE_END,
            # AND(stage_complete, NOT(has_blockers))
            NODE_START, Id(3), NOR, Id(0), Id(0), NODE_END,  # NOT(stage_complete)
            NODE_START, Id(4), NOR, Id(3), Id(2), NODE_END,  # AND
            OUTPUT_DECL, Id(4),
        ]
    },

    # Voting threshold (majority of 3)
    {
        "prompt": "Approve if at least 2 of 3 votes are yes (majority)",
        "description": "Majority vote: (a AND b) OR (b AND c) OR (a AND c)",
        "tokens": [
            INPUT_DECL, Id(0),  # vote_a
            INPUT_DECL, Id(1),  # vote_b
            INPUT_DECL, Id(2),  # vote_c
            # NOT(a), NOT(b), NOT(c) for AND operations
            NODE_START, Id(3), NOR, Id(0), Id(0), NODE_END,
            NODE_START, Id(4), NOR, Id(1), Id(1), NODE_END,
            NODE_START, Id(5), NOR, Id(2), Id(2), NODE_END,
            # a AND b
            NODE_START, Id(6), NOR, Id(3), Id(4), NODE_END,
            # b AND c
            NODE_START, Id(7), NOR, Id(4), Id(5), NODE_END,
            # a AND c
            NODE_START, Id(8), NOR, Id(3), Id(5), NODE_END,
            # (a AND b) OR (b AND c)
            NODE_START, Id(9), OR, Id(6), Id(7), NODE_END,
            # result OR (a AND c)
            NODE_START, Id(10), OR, Id(9), Id(8), NODE_END,
            OUTPUT_DECL, Id(10),
        ]
    },
]


def get_golden_examples() -> List[Dict]:
    """Get golden examples as training data format."""
    examples = []
    for golden in GOLDEN_EXAMPLES:
        # Skip examples with incomplete tokens (must have OUTPUT_DECL = 42)
        if OUTPUT_DECL not in golden["tokens"]:
            continue
        examples.append({
            "prompt": golden["prompt"],
            "completion": " ".join(str(t) for t in golden["tokens"])
        })
    return examples


def main():
    """Generate golden examples and write to JSONL."""
    examples = get_golden_examples()

    print(f"Generated {len(examples)} golden examples")

    with open("golden_examples.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print("\nGolden examples:")
    for example in examples:
        print(f"  {example['prompt']}")


if __name__ == "__main__":
    main()
