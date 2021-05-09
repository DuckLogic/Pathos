from __future__ import  annotations

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

@dataclass
class TokenType:
    name: str

@dataclass
class LiteralTokenType(TokenType):
    literal: str
    def __init__(self, literal: str):
        super().__init__(name='LITERAL')
        self.literal = literal

class Prefix(metaclass=ABCMeta):
    """Information on a list of literals that can prefix a case"""

    @abstractmethod
    def limit(self, length: int) -> Prefix:
        """Limit to the specified amount of items"""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

@dataclass(frozen=True, eq=True)
class LiteralPrefix(Prefix):
    tokens: tuple[TokenType]
    """The list of tokens that can prefix this item"""

    def limit(self, length: int) -> LiteralPrefix:
        assert length >= 1
        if length < len(self.tokens):
            return LiteralPrefix(self.tokens[:length])
        else:
            return self

@dataclass(frozen=True, eq=True)
class NegatedPrefix(Prefix):
    negated_token_set: frozenset[TokenType]
    """The set of literals that may **NOT** prefix this item"""

    def limit(self, length: int):
        assert length >= 1
        # We're always length one, so always fit under the limit
        return self


class AnalysedItem(metaclass=ABCMeta):
    @abstractmethod
    def list_prefixes(self, *, limit: int = 1) -> Iterator[tuple[TokenType]]:
        """List the possible prefixes that can start this item, up to the specified limit"""
        pass

@dataclass
class SpecificTokenItem(AnalysedItem):
    expected_token: TokenType

    def possible_starts(self, *, limit: int = 1):
        yield (self.expected_token,)

@dataclass
class AlternativeItemList(AnalysedItem):
    alternatives: list[AnalysedItem]

    def possible_starts(self, *, limit: int = 1):
        assert limit >= 1
        for alt in self.alternatives:
            yield from alt.possible_starts(limit=limit)

class AnalysedItem

class AnalysedCase:
    items: 

class AnalysedRule:
    cases: 
 
    def starts()