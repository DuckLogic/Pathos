from future import __annotations__

from metaclass import ABCMeta
from typing import Optional
from dataclasses import dataclass

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
    name = parser.expect(IDENT)
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
    name = parser.expect(IDENT)
    if name.peek() == "=":
        name.expect("=")
        rule = match_rule()
        return NamedRuleItem(name, rule)
    else:
        return RuleItem(name)

@dataclass
class MatchRule(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def parse(cls, parser) -> tuple[Optional[MatchRule], Optional[int]]:
        pass

@dataclass
class RepeatedMatchRule(MatchRule):
    repeated_rule: SimpleMatch
    repetition_type: Repetition

@dataclass
class OptionalMatchRule(MatchRule):
    optionally_matched: MatchRule

@dataclass
class NegativeAssertionMatchRule(MatchRule):
    literal: str

    @classmethod
    def parse(parser):
        parser.expect() 

def match_rule(parser):
    # match_rule[MatchRule]
    #     | '(' repeated_rule=simple_match ')' repetition_type -> Repeated { repeated_rule, repetition_type }
    #     | inner_rule=simple_match '?' -> Optional { rule: inner_rule }
    #     | negated_rule=negative_match -> Negative { negated_rule }
    #     
    #     | rule=simple_match -> Simple { rule }
    #     | '!' literal=LITERAL_STRING -> NegativeAssertion { literal };
    if parser.peek() == "(":
        # NOTE: This conflicts with multiple rules
        # In theory, it could be either ('(', repeated_rule),
        # or a simple_match for an optional or simple
        #
        # In practice, we don't care and are happy to ignore the
        # inner paranthesized 'simple_match'
        parser.expect('(')
        inner = simple_match()
        parser.expect(')')
        repetition = parser.attempt_parse(repetition_type)
        if repetition is not None:
            return RepeatedMatchRule(repetition_type=repetition, repeated_rule=inner)
        else:
            if parserr

    prediction = MATCH_RULE_PREDICTION_TABLE.get(parser.peek())
    if prediction is not None:
        return prediction(parser)
    else:
        # parse as an simple match
        inner_rule=simple_match()
        if parser.peek()




simple_match[SimpleMatchRule]: name=IDENT -> NamedRule { name }
    | text=LITERAL_STRING -> Literal { text }
    /* Ignore clarifying parentheses */
    | '(' inner=simple_match ')' -> nospan { inner };

repetition_type[RepetitionType]: separator=LITERAL_STRING? '*' -> ZeroOrMore { separator }
    | separator=LITERAL_STRING? '+' -> OneOrMore { separator }
    | separator=LITERAL_STRING '**' -> ZeroOrMoreTerminated { separator }
    | separator=LITERAL_STRING '++' -> OneOrMoreTerminated { separator }
    | '**' -> error { "Repetition with '**' requires a separator" }
    | '++' -> error { "Repetition with '++' requires a separator"};

/*
 * NOTE: There's a difference between a lookahead assertion
 * and a 'negative match'. A negative match like ^('{' | '}')
 * matches all tokens except '{' and '}'. A lookahead
 * assertion like !';' fails the case if a ';' is present,
 * but doesn't consume any tokens.
 */
negative_match[NegativeMatch]: '^' literal=LITERAL_STRING -> Literal { literal }
    | literals=(LITERAL_STRING)'|'* -> LiteralSet { literals };

type_decl[TypeDecl]: '[' text=raw_type ']' -> TypeDecl { text:  } 
raw_type[str]: name=IDENT -> { node.text }
    | '[' inner=raw_type ']' -> { node.text };

handler[Handler]: ignore_span=NOSPAN? code=code_block -> Explicit { code, ignore_span }
    | variant_name=IDENT '{' attrs=(auto_attribute)','** '}' -> Auto { variant_name, attrs }
    | ERROR message=code_block -> Error { message };

code_block[CodeBlock]: '{' text=raw_code '}' -> CodeBlock { text };
raw_code[str]: ^('{' | '}')* { node.text }
    | '{' inner=raw_code '}' { node.text };}

/*
 * There are three ways to generate an attribute for an AST node:
 * Node { plain }
 * Node { renamed: original }
 * Node { generaed: { arbitrary_code() } }
 */
auto_attribute[Attribute]: name=IDENT -> Named { name }
    | name=IDENT '=' original=IDENT -> Renamed { name, original: src }
    | name=IDENT '=' code=code_block -> RawCode { name, code };

