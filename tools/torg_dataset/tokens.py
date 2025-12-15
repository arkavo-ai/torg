"""TØR-G token definitions and Ministral mapping."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Union


class TokenType(IntEnum):
    """TØR-G token types with Ministral IDs."""
    OR = 36
    NOR = 37
    XOR = 38
    NODE_START = 39
    NODE_END = 40
    INPUT_DECL = 41
    OUTPUT_DECL = 42
    TRUE = 43
    FALSE = 44
    # Id tokens start at 45: Id(n) = 45 + n


@dataclass
class Token:
    """A TØR-G token with its Ministral ID."""
    type: TokenType
    id_value: int = None  # Only for Id tokens

    @property
    def ministral_id(self) -> int:
        """Get the Ministral token ID."""
        if self.id_value is not None:
            return 45 + self.id_value
        return int(self.type)

    def __repr__(self) -> str:
        if self.id_value is not None:
            return f"Id({self.id_value})"
        return self.type.name


# Convenience constructors
def Or() -> Token:
    return Token(TokenType.OR)

def Nor() -> Token:
    return Token(TokenType.NOR)

def Xor() -> Token:
    return Token(TokenType.XOR)

def NodeStart() -> Token:
    return Token(TokenType.NODE_START)

def NodeEnd() -> Token:
    return Token(TokenType.NODE_END)

def InputDecl() -> Token:
    return Token(TokenType.INPUT_DECL)

def OutputDecl() -> Token:
    return Token(TokenType.OUTPUT_DECL)

def True_() -> Token:
    return Token(TokenType.TRUE)

def False_() -> Token:
    return Token(TokenType.FALSE)

def Id(n: int) -> Token:
    assert 0 <= n <= 255, f"Id must be 0-255, got {n}"
    return Token(TokenType.INPUT_DECL, id_value=n)  # type is ignored for Id


# Fix Id token
def Id(n: int) -> Token:
    """Create an Id token."""
    assert 0 <= n <= 255, f"Id must be 0-255, got {n}"
    t = Token(TokenType.OR)  # Placeholder type, overridden by id_value
    t.id_value = n
    return t


def tokens_to_ministral_ids(tokens: List[Token]) -> List[int]:
    """Convert a list of TØR-G tokens to Ministral token IDs."""
    return [t.ministral_id for t in tokens]


def tokens_to_string(tokens: List[Token]) -> str:
    """Convert tokens to space-separated Ministral IDs."""
    return " ".join(str(t.ministral_id) for t in tokens)
