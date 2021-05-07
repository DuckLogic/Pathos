from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional
from dataclasses import dataclass

from pathos.runtime import Parser, Ident
from .lexer import LiteralString

@dataclass
class Grammar:
    rules: list[Rule]

def grammar(parser) -> Grammar:
    result = []
    while parser.peek() is not None:
        result.append(rule(parser))
    return Grammar(rules=rules)

@dataclass
class Rule:
    name: list[Ident]
    result_type: TypeDecl
    cases: list[RuleCase]

def rule(parser) -> Rule:
    name = parser.expect(Ident)
    result_type = parser.expect(type_decl)
    parser.expect(':')
    cases = parser.parse_repeated(rule_case, minimum=1, seperator='|')
    parser.expect(';')
    return Rule(name, result_type, cases)

@dataclass
class RuleCase:
    items: list[RuleItem]
    handler: Handler

def rule_case(parser):
    items = parser.parse_repeated(rule_item)
    print(items)
    parser.expect('->')
    return RuleCase(items, handler=parser.expect(handler))

@dataclass
class RuleItem:
    name: Ident

@dataclass
class NamedRuleItem(RuleItem):
    rule: MatchRule

def rule_item(parser) -> RuleItem:
    if isinstance(parser.peek(), Ident) \
        and parser.peek(ahead=1) == '=':
        name = parser.expect(Ident)
        parser.expect("=")
        rule = parser.expect(match_rule)
        return NamedRuleItem(name, rule)
    else:
        return parser.expect(match_rule)

@dataclass
class MatchRule(metaclass=ABCMeta):
    @classmethod
    def parse(cls, parser) -> tuple[Optional[MatchRule], Optional[int]]:
        raise NotImplementedError

@dataclass
class NegatedMatchRule(MatchRule):
    literals: frozenset[LiteralString]

    @classmethod
    def parse(cls, parser):
        parser.expect('^')
        literals: frozenset
        if parser.peek() == "(":
            parser.expect("(")
            literals = parser.parse_repeated(LiteralString, minimum=1, seperator="|")
            parser.expect(")")
        elif instanceof(parser.peek(), LiteralString):
            literals = frozenset((parser.expect(LiteralString),))
        else:
            raise parser.unexpected_token()
        return NegatedMatchRule(literals=literals)


@dataclass
class RepeatedMatchRule(MatchRule):
    repeated_rule: SimpleMatch
    repetition_type: Repetition

    @classmethod
    def parse(cls, parser):
        parser.expect('(')
        repeated_rule = parser.expect(simple_match)
        parser.expect(')')
        repetition = parser.expect(repetition_type)
        return RepeatedMatchRule(repeated_rule, repetition_type=repetition)

@dataclass
class OptionalMatchRule(MatchRule):
    inner_rule: MatchRule

@dataclass
class NegativeAssertionMatchRule(MatchRule):
    literal: LiteralString

    @classmethod
    def parse(cls, parser):
        parser.expect('!')
        literal = parser.expect(LiteralString)
        return NegativeAssertionMatchRule(literal=literal)

MATCH_RULE_TABLE = {
    '(': RepeatedMatchRule.parse,
    '^': NegatedMatchRule.parse,
    '!': NegativeAssertionMatchRule.parse
}
def match_rule(parser) -> MatchRule:
    # match_rule[MatchRule]
    #     | '(' repeated_rule=simple_match ')' repetition_type -> Repeated { repeated_rule, repetition_type }
    #     | inner_rule=simple_match '?' -> Optional { rule: inner_rule }
    #     | negated_rule=negative_match -> Negative { negated_rule }
    #     
    #     | rule=simple_match -> Simple { rule }
    #     | '!' literal=LITERAL_STRING -> NegativeAssertion { literal };
    parse_func = MATCH_RULE_TABLE.get(str(parser.peek()))
    if parse_func is not None:
        return parse_func(parser)
    else:
        # Must start with 'simple_match'
        inner_rule = simple_match(parser)
        if parser.peek() == "?":
            parser.expect('?')
            return OptionalMatchRule(inner_rule=inner_rule)
        else:
            return inner_rule

@dataclass
class NamedRule(MatchRule):
    name: Ident

@dataclass
class LiteralRule(MatchRule):
    literals: list[LiteralString]


def simple_match(parser) -> MatchRule:
    if isinstance(parser.peek(), Ident):
        return NamedRule(name=parser.expect(Ident))
    elif isinstance(parser.peek(), LiteralString):
        first = parser.expect(LiteralString)
        if parser.peek() == "|":
            parser.peek('|')
            others = parser.parse_repeated(LiteralString, minimum=1, seperator='|')
            return LiteralRule([first, *others])
        else:
            return LiteralRule(literals=[first])
    else:
        raise parser.unexpected_token()

@dataclass
class RepetitionType:
    seperator: Optional[str] = None
    minimum: int = 0
    allow_extra_terminator: bool = False

REPETITION_MARKERS: dict[str, int] = {
    '*': 0, '+': 1, '**': 0, '++': 1
}
def repetition_type(parser) -> RepetitionType:
    seperator = None
    if isinstance(parser.peek(), LiteralString):
        seperator = parser.expect(LiteralString)
    repetition_str = parser.peek()
    try:
        minimum = REPETITION_MARKERS[str(repetition_str)]
    except KeyError:
        raise parser.unexpected_token()
    else:
        parser.expect(repetition_str) # consume
    return RepetitionType(
        seperator=seperator, minimum=minimum,
        allow_extra_terminator=len(repetition_str.text) == 2
    )

@dataclass
class TypeDecl:
    text: str

def type_decl(parser):
    parser.expect('[')
    def raw_type_decl(parser) -> str:
        if isinstance(parser.peek(), Ident):
            return str(parser.expect(Ident))
        elif parser.peek() == '[':
            parser.expect('[')
            inner = raw_type_decl(parser)
            parser.expect(']')
            return f"[{inner}]"
        else:
            raise parser.unexpected_token()
    text=raw_type_decl(parser)
    parser.expect(']')
    return TypeDecl(text=text)

class Handler(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def parse(cls, parser):
        pass

@dataclass
class ExplicitHandler(Handler):
    code: CodeBlock
    ignore_span: bool = False

    @classmethod
    def parse(cls, parser):
        if parser.peek() == "nospan":
            parser.expect("nospan")
            ignore_span = True
        else:
            ignore_span = False
        code = code_block(parser)
        return ExplicitHandler(code=code, ignore_span=ignore_span)

@dataclass
class AutoHandler(Handler):
    variant_name: Ident
    attrs: list[AutoAttribute]

    @classmethod
    def parse(cls, parser):
        variant_name = parser.expect(Ident)
        parser.expect('{')
        attrs = parser.parse_repeated(
            auto_attribute, seperator=',',
            allow_extra_terminator=True
        )
        parser.expect('}')
        return AutoHandler(variant_name, attrs)

@dataclass
class ErrorHandler(Handler):
    message: CodeBlock

    @classmethod
    def parse(cls, parser):
        parser.expect("error")
        message = code_block(parser)
        return ErrorHandler(message)

HANDLER_PARSE_TABLE = {
    "nospan": ExplicitHandler.parse,
    '{': ExplicitHandler.parse,
    Ident: AutoHandler.parse,
    "error": ErrorHandler.parse
}
def handler(parser) -> Handler:
    parse_func = parser.predict(HANDLER_PARSE_TABLE)
    if parse_func is None:
        raise parser.unexpected_token()
    return parse_func(parser)

@dataclass
class CodeBlock:
    text: str

def code_block(parser) -> CodeBlock:
    def raw_code(parser) -> str:
        result = []
        while (token := parser.peek()) not in ("{", "}"):
            parser.expect(token)
            if isinstance(token, str):
                result.append(token)
            else:
                result.append(token.text)
        if result:
            return ''.join(result)
        else:
            if parser.peek() == "{":
                parser.expect("{")
                inner = raw_code(parser)
                parser.expect("}")
                return f"{{{inner}}}"
            else:
                raise parser.unexpected_token()
    parser.expect('{')
    inner = raw_code(parser)
    parser.expect('}')
    return CodeBlock(text=inner)

@dataclass
class AutoAttribute(metaclass=ABCMeta):
    name: str

class NamedAutoAttribute(AutoAttribute):
    pass

@dataclass
class RenamedAutoAttribute(AutoAttribute):
    original: Ident

@dataclass
class RawCodeAutoAttribute(AutoAttribute):
    code: CodeBlock

def auto_attribute(parser) -> AutoAttribute:
    name = parser.expect(Ident)
    if parser.peek() != ":":
        return NamedAutoAttribute(name=name)
    parser.expect(':')
    if isinstance(parser.peek(), Ident):
        original = parser.expect(Ident)
        return RenamedAutoAttribute(name=name, original=original)
    elif parser.peek() == "{":
        code = code_block(parser)
        return RawCodeAutoAttribute(name=name, code=code)
    else:
        raise parser.unexpected_token()
