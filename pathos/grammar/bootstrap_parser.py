from future import __annotations__

from metaclass import ABCMeta
from typing import Optional
from dataclasses import dataclass

from pathos.runtime import Parser, Ident

@dataclass
class Grammar:
    rules: list[Rule]

def grammar(parser) -> Grammar:
    return Grammar(rules=parser.parse_repeated(rule))

@dataclass
class Rule:
    name: list[Ident]
    result_type: TypeDecl
    cases: list[RuleCase]

def rule(parser) -> Rule:
    name = parser.expect(Ident)
    result_type = parser.expect(type_decl)
    parser.expect(':')
    cases = parser.parse_repeated(min=1, separator='|')
    return Rule(name, result_type, cases)

@dataclass
class RuleCase:
    items: list[RuleItem]
    handler: Handler

def rule_case(parser):
    items = parser.parse_repeated(rule_item)
    parser.expect('->')
    handler = parser.expect(handler)
    return RuleCase(items, handler)

@dataclass
class RuleItem:
    name: Ident

@dataclass
class NamedRuleItem(RuleItem):
    rule: MatchRule

def rule_item(parser) -> RuleItem:
    # NOTE: Both cases start with IDENT
    name = parser.expect(Ident)
    if name.peek() == "=":
        name.expect("=")
        rule = match_rule()
        return NamedRuleItem(name, rule)
    else:
        return RuleItem(name)

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
            literals = parser.parse_repeated(LiteralString, minimum=1, separator=",")
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
    parse_func = MATCH_RULE_TABLE.get(parser.peek())
    if parse_func is not None:
        return parse_func(parser)
    else:
        # Must start with 'simple_match'
        inner_rule = simple_match(parser)
        if parser.peek() == "?":
            return OptionalMatchRule(inner_rule=inner_rule)
        else:
            return inner_rule

class NamedRule(MatchRule):
    name: Ident

class LiteralRule(MatchRule):
    literal_text: LiteralString

def simple_match(parser) -> MatchRule:
    if isinstance(parser.peek(), Ident):
        return NamedRule(name=parser.expect(Ident))
    elif isinstance(parser.peek(), LiteralString):
        return NamedRule(literal_text=parser.expect(LiteralString))
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
    marker_str 
    seperator = None
    if isinstance(parser.peek(), LiteralString):
        seperator = parser.expect(LiteralString)
    repetition_str = parser.peek()
    try:
        minimum = REPETITION_MARKERS[repetition_str]
    except KeyError:
        raise parser.unexpected_token()
    else:
        parser.expect(repetition_str) # consume
    return RepetitionType(
        seperator=separator, minimum=minimum,
        allow_extra_terminator=len(repetition_str) == 2
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

class ExplicitHandler(Handler):
    code: CodeBlock
    ignore_span: bool = False

    @classmethod
    def parse(cls, parser):
        if pasrer.peek() == "nospan":
            parser.expect("nospan")
            ignore_span = True
        else:
            ignore_span = False
        code = code_block(parser)
        return ExplicitHandler(code=code, ignore_span=ignore_span)

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
    parser.parse('}')
    return CodeBlock(text=inner)

class AutoAttribute(metaclass=ABCMeta):
    name: str

class NamedAutoAttribute(AutoAttribute):
    pass

class RenamedAutoAttribute(AutoAttribute):
    original: Ident

class RawCodeAutoAttribute(AutoAttribute):
    code: CodeBlock

def auto_attribute(parser) -> AutoAttribute:
    name = parser.expect(Ident)
    if parser.peek() != "=":
        return NamedAutoAttribute(name=name)
    parser.expect('=')
    if isinstance(parser.peek(), Ident):
        original = parser.expect(Ident)
        return AutoAttribute(name=name, original=original)
    elif parser.peek() == "{":
        code = code_block(parser)
        return RawCodeAutoAttribute(name=name, code=code)
    else:
        raise parser.unexpected_token()
