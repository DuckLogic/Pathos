from __future__ import  annotations

import sys

from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from .grammar import bootstrap_parser as parser
from .runtime import Ident
from .utils import pairwise_longest, gv_escape

@dataclass(frozen=True, eq=True)
class TokenType:
    name: str

    def __eq__(self):
        if type(other) is TokenType:
            return self.name == other.name
        else:
            return NotImplemented

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

@dataclass(frozen=True, eq=True)
class LiteralTokenType(TokenType):
    literal: str
    def __init__(self, literal: str):
        super().__init__(name='LITERAL')
        object.__setattr__(self, 'literal', literal)

    def __eq__(self, other):
        if type(other) is LiteralTokenType:
            return self.literal == other.literal
        elif isinstance(other, TokenType):
            return False
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.literal)

    def __str__(self):
        return repr(self.literal)

class Prefix(metaclass=ABCMeta):
    """Information on a list of literals that can prefix a case"""

    @abstractmethod
    def limit(self, length: int) -> Prefix:
        """Limit to the specified amount of items"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

@dataclass(frozen=True, eq=True)
class LiteralPrefix(Prefix):
    tokens: tuple[TokenType, ...]
    """The list of tokens that can prefix this item"""

    def limit(self, length: int) -> LiteralPrefix:
        assert length >= 1
        if length < len(self.tokens):
            return LiteralPrefix(self.tokens[:length])
        else:
            return self

    def __add__(self, other):
        if isinstance(other, LiteralPrefix):
            return LiteralPrefix(tokens=self.tokens + other.tokens)
        else:
            return NotImplemented

    def __len__(self):
        return len(self.tokens)

    def __str__(self) -> str:
        return ', '.join(map(str, self.tokens))

@dataclass(frozen=True, eq=True)
class NegatedPrefix(Prefix):
    negated_token_set: frozenset[TokenType]
    """The set of literals that may **NOT** prefix this item"""

    def limit(self, length: int):
        assert length >= 1
        # We're always length one, so always fit under the limit
        return self

    def __len__(self):
        return 1

    def __str__(self):
        tokens = sorted(self.negated_token_set)
        return f"!({' | '.join(map(str, tokens))})"

@dataclass(frozen=True, eq=True)
class EmptyPrefix(Prefix):
    def limit(self, length: int):
        assert length >= 0
        # We're always length zero, so always fit under the limit
        return self

    def __add__(self, other):
        if isinstance(other, Prefix):
            return other
        else:
            return NotImplemented

    def __len__(self):
        return 0

    def __str__(self):
        return "EMPTY"

class PrefixSet:
    """A set of possible prefixes"""
    __slots__ = "prefixes"
    prefixes: frozenset[Prefix]

    def __init__(self, prefixes):
        self.prefixes = frozenset(prefixes)

    def __hash__(self) -> int:
        return hash(self.prefixes)

    def __eq__(self, other):
        if isinstance(other, PrefixSet):
            return self.prefixes == other.prefixes
        else:
            return NotImplemented

    def __len__(self):
        return len(self.prefixes)

    def __iter__(self):
        return iter(self.prefixes)

    def __repr__(self) -> str:
        return f"PrefixSet({self.prefixes})"

_num_ids = 0
@dataclass
class AnalysedItem(metaclass=ABCMeta):
    name: Optional[str]

    def __post_init__(self):
        assert self.name is None or isinstance(self.name, str), \
            f"Invalid type of name: {self.name!r}"

    @property
    def id(self) -> str:
        global _num_ids
        if (name := self.name) is not None:
            return name
        self.id = _num_ids
        _num_ids += 1
        return f"n{self.id}"

    @abstractmethod
    def print_graph(self) -> Iterator[str]:
        yield from ()

    @abstractmethod
    def list_prefixes(self, length: int = None) -> Iterator[Prefix]:
        yield from ()

PrefixMap: TypeAlias = dict[Prefix, list[AnalysedItem]]

@dataclass
class SimpleItem(AnalysedItem):
    tokens: tuple[TokenType, ...]

    @property
    def prefix(self) -> LiteralPrefix:
        return LiteralPrefix(tuple(self.tokens))

    def print_graph(self) -> str:
        if getattr(self, "printed", False):
            return
        self.printed = True
        yield f"{self.id} [label={gv_escape(str(self.prefix))}];"

    def list_prefixes(self, length: int = None):
        res = self.prefix
        if length is not None:
            yield self.prefix.limit(length)
        else:
            yield self.prefix

@dataclass
class ChainedItem(AnalysedItem):
    chained: list[AnalysedItem]

    def print_graph(self):
        if getattr(self, 'printed', False):
            return
        self.printed = True
        for item in self.chained:
            yield from item.print_graph()
        for item, next_item in pairwise_longest(self.chained):
            if next_item is not None:
                yield f'{item.id} -> {next_item.id} [label="next"];'

    def list_prefixes(self, length: int = None):
        # TODO: How do we handle combinatorial blowup?
        if len(self.chained) > 2:
            first, second = self.chained[0], \
                ChainedItem(name=None, chained=self.chained[1:])
        elif len(self.chained) == 2:
            first, second = self.chained
        elif len(self.chained) == 1:
            yield from self.chained.list_prefixes(length=length)
        else:
            return
        for first_prefix in first.list_prefixes(length=length):
            if len(first_prefix) >= length:
                # We just need first
                yield first_prefix
                continue
            # Combinations with second
            for second in second.list_prefixes(length=length-len(first_prefix)):
                try:
                    yield first_prefix + second
                except TypeError:
                    pass  # Can't add these I guess




@dataclass
class AlternativeItemList(AnalysedItem):
    alternatives: list[AnalysedItem]

    def disambiguate_prefixes(self) -> tuple[int, dict[Prefix, list[AnalysedItem]]]:
        max_length = 1
        ambiguous = dict()
        res = defaultdict(list)
        while True:
            ambiguous.clear()
            res.clear()
            all_are_longest = True
            for alt in self.alternatives:
                prefixes = PrefixSet(alt.list_prefixes(length=max_length))
                for prefix in prefixes:
                    is_longest = len(prefixes) < max_length
                    if not is_longest:
                        all_are_longest = False
                    existing_alts = res[prefix]
                    existing_alts.append(alt)
                    if len(existing_alts) > 1:
                        existing_alts.append(alt)
                        # We already have another alternative with this prefix
                        # Possible ambiguity (at least at this length)
                        ambiguous[prefix] = existing_alts
            # If we have any ambiguity, and are not at our longest
            # length, then we should try again.
            if ambiguous and not all_are_longest:
                max_length += 1
            else:
                break
        return max_length, res

    def print_graph(self):
        if getattr(self, 'printed', False):
            return
        self.printed = True
        max_length, disambiguated = self.disambiguate_prefixes()
        label = f"{self.name} (cplx)" if self.name is not None else f"{self.id} (complex)"
        yield f"{self.id} [label={gv_escape(label)}];"
        for alt in self.alternatives:
            yield from alt.print_graph()
        for prefixes, alts in disambiguated.items():
            for alt in alts:
                yield f"{self.id} -> {alt.id} [label={gv_escape(str(prefixes))}];"


    def list_prefixes(self, length: int = None):
        unique = set()
        for alt in self.alternatives:
            prefixes = list(alt.list_prefixes(length=length))
            unique.update(prefixes)
        yield from unique

@dataclass
class RepeatedItem(AnalysedItem):
    repetition_type: RepetitionType
    seperator_token: Optional[TokenType]
    repeated_item: AnalysedItem
    
    def print_graph(self):
        if getattr(self, 'printed', False):
            return
        self.printed = True
        label = f"{self.id}{self.repetition_type.char}"
        yield f"{self.id} [label={gv_escape(label)}];"
        yield from self.repeated_item.print_graph()
        yield f'{self.id} -> {self.repeated_item.id} [label="first"];'
        if self.seperator_token is not None:
            next_label = str(self.seperator_token)
        else:
            next_label = "try_next"
        yield f"{self.repeated_item.id} -> {self.id} [label={gv_escape(next_label)}];"

    def list_prefixes(self, length: int = None):
        if self.repetition_type.minimum == 0:
            yield EmptyPrefix()
        # Only give one repetition worth of prefixes
        for prefix in self.repeated_item.list_prefixes(length=length):
            if len(prefix) < length and self.seperator_token is not None:
                # Try and add our prefix at the end
                try:
                    prefix = prefix + LiteralPrefix(tokens=(self.seperator_token,))
                except TypeError:
                    # Its fine if we cant do it though
                    pass
            yield prefix

class AnalysedRule:
    original: parser.Rule
    name: Ident
    item: AlternativeItemList
    """The rule as an analysed item"""

    def __init__(self, original: parser.Rule):
        self.original = original
        self.name = original.name
        self.item = AlternativeItemList(
            name=str(self.name),
            alternatives=[]  # lazy init
        )

class AnalysedGrammar:
    original: parser.Grammar
    token_types: list[TokenType]
    rules: dict[str, AnalysedRule]
    named_token_types: dict[str, TokenType]

    def __init__(self, original, token_types):
        self.original = original
        self.token_types = token_types
        self.rules = {}
        for orig_rule in original.rules:
            if orig_rule.name in self.rules:
                raise ValueError(f"Duplicate rules for name {rule.name}")
            self.rules[orig_rule.name] = AnalysedRule(
                original=orig_rule
            )
        self.named_token_types = {}
        for token in self.token_types:
            if isinstance(token, LiteralTokenType):
                continue
            self.named_token_types[token.name] = token

    def analyse(self):
        for orig_rule in self.original.rules:
            rule = self.rules[str(orig_rule.name)] 
            assert not rule.item.alternatives, \
                "Alternatives should be empty"
            for case_index, orig_case in enumerate(orig_rule.cases):
                case_name = f"{orig_rule.name}_{case_index}"
                if not orig_case.items:
                    raise ValueError(f"No items for case {case_name}")
                elif len(orig_case.items) == 1:
                    case_item = self.analyse_item(
                        orig_case.items[0],
                        name=case_name
                    )
                else:
                    case_item = ChainedItem(
                        chained=[
                            self.analyse_item(
                                sub_item,
                                name=f"{case_name}_{sub_index}"
                            ) for sub_index, sub_item in \
                            enumerate(orig_case.items)
                        ],
                        name=case_name
                    )
                rule.item.alternatives.append(case_item)

    def analyse_item(self, item: parser.RuleItem, name: str = None) -> AnalysedItem:
        if isinstance(item, parser.NamedRuleItem):
            return self.analyse_match(item.rule, name=str(item.name))
        elif isinstance(item, parser.MatchRule):
            return self.analyse_match(item, name=name)
        else:
            raise TypeError(f"Unexpected type for item: {type(item)!r}")

    def analyse_match(self, item: parser.MatchRule, *, name: str = None) -> AnalysedItem:
        assert isinstance(item, parser.MatchRule), \
            f"Expected a match rule: {item!r}"
        if isinstance(item, parser.LiteralRule):
            return SimpleItem(
                name=name,
                tokens=[LiteralTokenType(lit.value) for lit in item.literals]
            )
        elif isinstance(item, parser.NamedRule):
            item_name = str(item.name)
            if item_name in self.named_token_types:
                return SimpleItem(
                    name=name,
                    tokens=[self.named_token_types[item_name]]
                )
            try:
                return self.rules[item_name].item
            except KeyError:
                raise ValueError(f"Unknown rule/token name {item.name!r}")
        elif isinstance(item, parser.NegatedMatchRule):
            raise NotImplementedError("Negated matches")
        elif isinstance(item, parser.RepeatedMatchRule):
            return RepeatedItem(
                name=name,
                repetition_type=item.repetition_type,
                seperator_token=LiteralTokenType(item.repetition_type.seperator.value)
                    if item.repetition_type.seperator is not None else None,
                repeated_item=self.analyse_match(item.repeated_rule)
            )
        elif isinstance(item, parser.OptionalMatchRule):
            raise NotImplementedError("Optional matches")
        elif isinstance(item, parser.NegativeAssertionMatchRule):
            raise NotImplementedError("Negative assertion match rules")
        else:
            raise TypeError(f"Unexpected item type {type(item).__name__!r}: {item!r}")

if __name__ == "__main__":
    grammar = parser.cmd_parse()
    try:
        token_types = []
        with open(sys.argv[2]) as token_file:
            from .grammar.lexer import is_valid_ident
            import ast
            for line_index, line in enumerate(token_file):
                line = line.strip()
                if is_valid_ident(line):
                    # Parse as ident
                    token_types.append(TokenType(name=line))
                    continue
                else:
                    try:
                        literal = ast.literal_eval(line)
                        if isinstance(literal, str):
                            token_types.append(LiteralTokenType(literal))
                            continue
                        else:
                            raise SyntaxError(f"Unexpected literal type {literal!r}")
                    except SyntaxError as e:
                        print(f"Syntax error parsing line #{line_index + 1}:", e, file=sys.stderr)
                        sys.exit(1)
                token_types.append(literal)
    except IOError:
        print("Unable to parse token file (second arg)", file=sys.stderr)
        sys.exit(1)
    except IndexError:
        print("Please specify two args: <grammar file> <token file>", file=sys.stderr)
        sys.exit(1)
    analysed = AnalysedGrammar(grammar, token_types=token_types)
    print("Analyzing grammar...", file=sys.stderr)
    analysed.analyse()
    print("Printing graph:", file=sys.stderr)
    print("digraph llk {")
    for rule in analysed.rules.values():
        for line in rule.item.print_graph():
            print('  ' + line)
    print("}")
