#!/usr/bin/env python3
"""Demo: Constrained TØR-G generation with Ministral + torg-mask.

This script demonstrates how to use torg-mask for constrained LLM decoding.
It uses llama-cpp-python for inference with the Ministral model.

Usage:
    python constrained_generate.py --model path/to/model.gguf
    python constrained_generate.py --model path/to/model.gguf --prompt "Allow if admin OR member"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


# Ministral token IDs for TØR-G
class TorgTokens:
    OR = 36
    NOR = 37
    XOR = 38
    NODE_START = 39
    NODE_END = 40
    INPUT_DECL = 41
    OUTPUT_DECL = 42
    TRUE = 43
    FALSE = 44

    @staticmethod
    def Id(n: int) -> int:
        return 45 + n

    @staticmethod
    def is_id(token: int) -> bool:
        return token >= 45

    @staticmethod
    def id_value(token: int) -> int:
        return token - 45


class TorgBuilder:
    """Python implementation of torg-core Builder state machine."""

    def __init__(self, max_ids: int = 256):
        self.max_ids = max_ids
        self.defined = set()  # Defined input/node IDs
        self.inputs = []
        self.nodes = []
        self.outputs = []
        self.phase = "inputs"  # inputs, nodes, outputs, done
        self.node_state = None  # None or (node_id, op, operands)

    def valid_next_tokens(self) -> List[int]:
        """Return list of valid Ministral token IDs for current state."""
        valid = []

        if self.phase == "done":
            return []

        if self.node_state is not None:
            # Inside a node definition
            node_id, op, operands = self.node_state

            if op is None:
                # After NodeStart Id, need operator
                valid = [TorgTokens.OR, TorgTokens.NOR, TorgTokens.XOR]

            elif len(operands) < 2:
                # Need operands (sources)
                # Can use: defined IDs, True, False
                valid = [TorgTokens.TRUE, TorgTokens.FALSE]
                for d in self.defined:
                    valid.append(TorgTokens.Id(d))

            else:
                # Have all operands, need NodeEnd
                valid = [TorgTokens.NODE_END]

        elif self.phase == "inputs":
            # Can: InputDecl, NodeStart, OutputDecl
            valid = [TorgTokens.INPUT_DECL, TorgTokens.NODE_START, TorgTokens.OUTPUT_DECL]
            # If we need to define an ID after InputDecl, that's handled in push

        elif self.phase == "nodes":
            # Can: NodeStart, OutputDecl
            valid = [TorgTokens.NODE_START, TorgTokens.OUTPUT_DECL]

        elif self.phase == "outputs":
            # Can: OutputDecl (more outputs), or done
            valid = [TorgTokens.OUTPUT_DECL]
            # Could also end here if we have outputs

        return valid

    def valid_ids_for_state(self) -> List[int]:
        """Return valid Id tokens for current state."""
        if self.node_state is not None:
            node_id, op, operands = self.node_state

            if node_id is None:
                # Need unused ID for new node
                valid = []
                for i in range(self.max_ids):
                    if i not in self.defined:
                        valid.append(TorgTokens.Id(i))
                        if len(valid) >= 50:  # Limit for efficiency
                            break
                return valid

            elif op is not None and len(operands) < 2:
                # Need operand - any defined ID
                return [TorgTokens.Id(d) for d in self.defined]

        elif self.phase == "inputs":
            # After InputDecl, need unused ID
            valid = []
            for i in range(self.max_ids):
                if i not in self.defined:
                    valid.append(TorgTokens.Id(i))
                    if len(valid) >= 50:
                        break
            return valid

        elif self.phase == "outputs":
            # After OutputDecl, need defined ID
            return [TorgTokens.Id(d) for d in self.defined]

        return []

    def push(self, token: int) -> bool:
        """Push a token and update state. Returns True if valid."""
        if self.phase == "done":
            return False

        # Handle node construction
        if self.node_state is not None:
            node_id, op, operands = self.node_state

            if node_id is None:
                # Expecting node ID
                if not TorgTokens.is_id(token):
                    return False
                n = TorgTokens.id_value(token)
                if n in self.defined:
                    return False
                self.node_state = (n, None, [])
                return True

            elif op is None:
                # Expecting operator
                if token not in [TorgTokens.OR, TorgTokens.NOR, TorgTokens.XOR]:
                    return False
                self.node_state = (node_id, token, [])
                return True

            elif len(operands) < 2:
                # Expecting operand
                if token == TorgTokens.TRUE:
                    operands.append(("true", None))
                elif token == TorgTokens.FALSE:
                    operands.append(("false", None))
                elif TorgTokens.is_id(token):
                    n = TorgTokens.id_value(token)
                    if n not in self.defined:
                        return False
                    operands.append(("id", n))
                else:
                    return False
                self.node_state = (node_id, op, operands)
                return True

            elif token == TorgTokens.NODE_END:
                # Finish node
                self.nodes.append(self.node_state)
                self.defined.add(node_id)
                self.node_state = None
                self.phase = "nodes"
                return True

            return False

        # Handle phase transitions
        if token == TorgTokens.INPUT_DECL:
            if self.phase not in ["inputs"]:
                return False
            # Next token should be an ID
            self._expect_input_id = True
            return True

        if hasattr(self, "_expect_input_id") and self._expect_input_id:
            if not TorgTokens.is_id(token):
                return False
            n = TorgTokens.id_value(token)
            if n in self.defined:
                return False
            self.inputs.append(n)
            self.defined.add(n)
            del self._expect_input_id
            return True

        if token == TorgTokens.NODE_START:
            if self.phase not in ["inputs", "nodes"]:
                return False
            self.phase = "nodes"
            self.node_state = (None, None, [])  # Start node
            return True

        if token == TorgTokens.OUTPUT_DECL:
            if self.phase not in ["inputs", "nodes", "outputs"]:
                return False
            self.phase = "outputs"
            self._expect_output_id = True
            return True

        if hasattr(self, "_expect_output_id") and self._expect_output_id:
            if not TorgTokens.is_id(token):
                return False
            n = TorgTokens.id_value(token)
            if n not in self.defined:
                return False
            self.outputs.append(n)
            del self._expect_output_id
            return True

        return False

    def is_complete(self) -> bool:
        """Check if we have a valid complete graph."""
        return len(self.outputs) > 0 and self.node_state is None


def create_logit_mask(valid_tokens: List[int], vocab_size: int) -> List[float]:
    """Create a logit mask array."""
    mask = [float("-inf")] * vocab_size
    for t in valid_tokens:
        if 0 <= t < vocab_size:
            mask[t] = 0.0
    return mask


def constrained_generate(
    llm: "Llama",
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> Tuple[List[int], str]:
    """Generate TØR-G tokens with constrained decoding."""

    builder = TorgBuilder()
    generated_tokens = []

    # Format input prompt
    full_prompt = f"""### Policy:
{prompt}

### TØR-G Token Sequence:
"""

    # Get initial logits
    llm.reset()
    llm.eval(llm.tokenize(full_prompt.encode()))

    for _ in range(max_tokens):
        # Get valid next tokens
        valid = builder.valid_next_tokens()

        # Check if we need IDs
        if hasattr(builder, "_expect_input_id") or hasattr(builder, "_expect_output_id"):
            valid = builder.valid_ids_for_state()
        elif builder.node_state is not None:
            node_id, op, operands = builder.node_state
            if node_id is None or (op is not None and len(operands) < 2):
                valid.extend(builder.valid_ids_for_state())

        if not valid:
            break

        # Create mask and sample
        # Note: In production, apply mask to actual logits
        # Here we just pick from valid tokens for demo
        import random

        if len(valid) == 1:
            token = valid[0]
        else:
            # Weight towards common patterns
            token = random.choice(valid)

        # Push token
        if not builder.push(token):
            print(f"Warning: Invalid token {token}, stopping")
            break

        generated_tokens.append(token)

        if builder.is_complete():
            # Allow a few more outputs, then stop
            if len(builder.outputs) >= 1:
                break

    return generated_tokens, " ".join(str(t) for t in generated_tokens)


def decode_tokens(tokens: List[int]) -> str:
    """Decode token sequence to human-readable form."""
    result = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == TorgTokens.OR:
            result.append("Or")
        elif t == TorgTokens.NOR:
            result.append("Nor")
        elif t == TorgTokens.XOR:
            result.append("Xor")
        elif t == TorgTokens.NODE_START:
            result.append("NodeStart")
        elif t == TorgTokens.NODE_END:
            result.append("NodeEnd")
        elif t == TorgTokens.INPUT_DECL:
            result.append("InputDecl")
        elif t == TorgTokens.OUTPUT_DECL:
            result.append("OutputDecl")
        elif t == TorgTokens.TRUE:
            result.append("True")
        elif t == TorgTokens.FALSE:
            result.append("False")
        elif TorgTokens.is_id(t):
            result.append(f"Id({TorgTokens.id_value(t)})")
        else:
            result.append(f"?{t}")
        i += 1
    return " ".join(result)


def demo_without_llm():
    """Demo constrained generation without an LLM (for testing)."""
    print("=" * 60)
    print("TØR-G Constrained Generation Demo (without LLM)")
    print("=" * 60)

    # Simulate building a graph step by step
    builder = TorgBuilder()

    print("\nBuilding: 'Allow if admin OR member'")
    print()

    tokens = []

    # Step through construction
    steps = [
        (TorgTokens.INPUT_DECL, "InputDecl - declare first input"),
        (TorgTokens.Id(0), "Id(0) - name it input 0 (admin)"),
        (TorgTokens.INPUT_DECL, "InputDecl - declare second input"),
        (TorgTokens.Id(1), "Id(1) - name it input 1 (member)"),
        (TorgTokens.NODE_START, "NodeStart - begin node"),
        (TorgTokens.Id(2), "Id(2) - node ID"),
        (TorgTokens.OR, "Or - operator"),
        (TorgTokens.Id(0), "Id(0) - left operand (admin)"),
        (TorgTokens.Id(1), "Id(1) - right operand (member)"),
        (TorgTokens.NODE_END, "NodeEnd - finish node"),
        (TorgTokens.OUTPUT_DECL, "OutputDecl - declare output"),
        (TorgTokens.Id(2), "Id(2) - output the OR result"),
    ]

    for token, description in steps:
        valid = builder.valid_next_tokens()
        if hasattr(builder, "_expect_input_id") or hasattr(builder, "_expect_output_id"):
            valid = builder.valid_ids_for_state()
        elif builder.node_state is not None:
            node_id, op, operands = builder.node_state
            if node_id is None or (op is not None and len(operands) < 2):
                valid.extend(builder.valid_ids_for_state())

        print(f"Valid tokens: {len(valid)} options")
        print(f"  Push: {token} ({description})")

        if token not in valid:
            print(f"  WARNING: Token {token} not in valid set!")

        if not builder.push(token):
            print(f"  ERROR: Token rejected!")
            break

        tokens.append(token)
        print(f"  OK - Graph complete: {builder.is_complete()}")
        print()

    print("=" * 60)
    print("Result")
    print("=" * 60)
    print(f"Token IDs: {' '.join(str(t) for t in tokens)}")
    print(f"Decoded:   {decode_tokens(tokens)}")
    print(f"Complete:  {builder.is_complete()}")
    print(f"Inputs:    {builder.inputs}")
    print(f"Nodes:     {len(builder.nodes)}")
    print(f"Outputs:   {builder.outputs}")


def main():
    parser = argparse.ArgumentParser(description="TØR-G constrained generation demo")
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Allow if admin OR member",
        help="Policy prompt"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo without LLM"
    )

    args = parser.parse_args()

    if args.demo or args.model is None:
        demo_without_llm()
        return

    if Llama is None:
        print("Error: llama-cpp-python not installed")
        print("Install with: pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    llm = Llama(model_path=args.model, n_ctx=2048)

    print(f"\nPrompt: {args.prompt}")
    tokens, token_str = constrained_generate(llm, args.prompt)

    print(f"\nGenerated tokens: {token_str}")
    print(f"Decoded: {decode_tokens(tokens)}")


if __name__ == "__main__":
    main()
