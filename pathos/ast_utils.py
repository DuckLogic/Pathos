from contextlib import contextmanager
from functools import singledispatch
import ast
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import sys

ConstantType = Union[str, bool, int, float, None]

COLLECTION_CHAR_MAP = {
    dict: ('{', '}'),
    list: ('[', ']'),
    tuple: ('(', ')')
}
class PrettyDumper:
    def __init__(self, indent_size: int = 4):
        self.lines = []
        self._pending_line = []
        self._current_indent = 0
        self.indent_size = indent_size

    @contextmanager
    def indent(self):
        self._current_indent += 1
        yield
        self._current_indent -= 1

    def write_line(self, line: str = ""):
        if '\n' in line:
            self.write(line + '\n')
            return
        line = ''.join(self._pending_line) + line
        self._pending_line.clear()
        if line:
            indent = ' ' * (self._current_indent * self.indent_size)
            self.lines.append(indent + line)
        else:
            self.lines.append("")

    def write(self, text: str):
        if '\n' in text:
            if self._pending_line:
                text = ''.join(self._pending_line) + text
                self._pending_line.clear()
            index = 0
            while (next_newline := text.find('\n', index)) >= 0:
                self.write_line(text[index:next_newline])
                index = next_newline + 1
            if index < len(text):
                self._pending_line.append(text[index:])
        else:
            self._pending_line.append(text)

    def pretty_dump(
            self, node: Union[ast.AST, list, tuple, ConstantType], *, newline: bool = False,
            actually_object: bool = False
    ):
        if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
            self.write(repr(OP_TABLE[type(node)]))
        elif isinstance(node, ast.expr_context):
            self.write(repr(EXPR_CONTEXT_TABLE[type(node)]))
        elif isinstance(node, ast.Constant):
            self.write(repr(node.value))
        elif isinstance(node, ast.AST):
            field_names = node._fields
            self.write(type(node).__name__)
            data = {name: getattr(node, name) for name in field_names}
            self.pretty_dump(data, actually_object=True)
        elif isinstance(node, (dict, list, tuple)):
            opening_char, closing_char = COLLECTION_CHAR_MAP[type(node)]
            if actually_object:
                self.write('(')
            else:
                self.write(opening_char)
            if len(node) in (0, 1):
                # Single line
                if isinstance(node, dict):
                    for key, val in node.items():
                        if actually_object:
                            self.write(str(key))
                            self.write('=')
                        else:
                            self.pretty_dump(key)
                            self.write(': ')
                        self.pretty_dump(val)
                else:
                    for val in node:
                        self.pretty_dump(val)
            else:
                # Split across multiple lines
                self.write_line()
                with self.indent():
                    if isinstance(node, dict):
                        for key, val in node.items():
                            if actually_object:
                                self.write(key)
                                self.write('=')
                            else:
                                self.pretty_dump(key)
                                self.write(': ')
                            self.pretty_dump(val)
                            self.write_line(',')
                    else:
                        for val in node:
                            self.pretty_dump(val)
                            self.write_line(',')
            if actually_object:
                self.write(')')
            else:
                self.write(closing_char)
        else:
            self.write(repr(node))
        # Either way, consider writing a newline
        if newline:
            self.write_line()

    def __str__(self):
        res = '\n'.join(self.lines)
        return res + ''.join(self._pending_line)


class PrettyPrinter:
    maximum_length: int
    indent_size: int
    current_line: list[str]
    lines: list[str]

    def __init__(self, *, maximum_length: int = 100, indent_size: int = 4):
        self.maximum_length = maximum_length
        self.indent_size = indent_size
        self._current_indent = 4
        self.current_line = []
        self.lines = []

    @property
    def current_offset(self) -> int:
        return len(self.current_line)

    @property
    def should_split_line(self) -> bool:
        remaining_chars = self.maximum_length - len(self.current_line)
        return remaining_chars <= 20

    def write_line(self, /, s: str = None):
        self.write(s or "", newline=True)

    def write(self, s: str, *, newline: bool = False):
        assert isinstance(s, str)
        if '\n' in s:
            index = 0
            while (next_newline := s.find('\n', index)) >= 0:
                self.write(s[index:next_newline], newline=True)
                index = next_newline + 1
            if index + 1 != len(s):
                self.write(s[index:])
        else:
            self.current_line.extend(s)
            if newline:
                assert all(len(c) == 1 for c in self.current_line)
                self.lines.append(''.join(self.current_line))
                self.current_line.clear()

    def __str__(self) -> str:
        lines = self.lines.copy()
        if self.current_line:
            lines.append(''.join(self.current_line))
        return '\n'.join(lines)


class Expr(metaclass=ABCMeta):
    @abstractmethod
    def pretty_print(self, printer: PrettyPrinter):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

@dataclass(frozen=True)
class ConstantExpr(Expr):
    value: Union[int, str, float, bool, bytes, None]

    def pretty_print(self, printer: PrettyPrinter):
        printer.write(str(self))

    def __str__(self) -> str:
        return repr(self.value)

@dataclass
class SExpr(Expr):
    op: str
    args: list[Expr]

    def pretty_print(self, printer: PrettyPrinter):
        line_offset = None
        def indent_appropriately():
            assert line_offset is not None
            needed_indent = line_offset // printer.indent_size
            printer.write(" " * (printer.indent_size * needed_indent))
        def maybe_break_line():
            if printer.should_split_line:
                current_offset = printer.current_offset
                printer.write_line()
                indent_appropriately()
                return True
            else:
                return False
        printer.write('(')
        maybe_break_line()
        printer.write(self.op)
        if not maybe_break_line():
            printer.write(' ')
        for index, val in enumerate(self.args):
            at_end = index + 1 == len(self.args)
            val.pretty_print(printer)
            if not maybe_break_line():
                if not at_end:
                    printer.write(' ')
        printer.write(')')


    def __str__(self):
        return f"({self.op} {' '.join(map(str, self.args))})"

EXPR_CONTEXT_TABLE = {
    ast.Load: "load",
    ast.Del: "del",
    ast.Store: "store"
}
def expr_context(ctx: ast.expr_context) -> Optional[str]:
    assert isinstance(ctx, ast.expr_context)
    if isinstance(ctx, ast.Load):
        return None
    else:
        return EXPR_CONTEXT_TABLE[type(ctx)]

OP_TABLE = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.MatMult: '@',
    ast.Pow: '**',
    ast.Div: '/',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.Mod: '%',
    ast.FloorDiv: '//',
    ast.And: 'and',
    ast.Or: 'or',
    ast.BitAnd: '&',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Is: 'is',
    ast.IsNot: 'is not',
    ast.In: 'in',
    ast.NotIn: 'not in',
    ast.Invert: '~',
    ast.Not: 'not',
    ast.UAdd: '+',
    ast.USub: '-'
}
def op(op: Union[ast.operator, ast.unaryop, ast.boolop, ast.cmpop]):
    try:
        return OP_TABLE[type(op)]
    except KeyError:
        raise ValueError(f"Unknown op: {op!r}")


@singledispatch
def convert_expr(expr: ast.Constant) -> Expr:
    return ConstantExpr(expr.value)

@convert_expr.register
def _(expr: ast.Expression) -> Expr:
    return convert_expr(expr.body)


@convert_expr.register
def _(expr: ast.BinOp) -> Expr:
    return SExpr(op(expr.op), [convert_expr(expr.left), convert_expr(expr.right)])


@convert_expr.register
def _(expr: ast.UnaryOp) -> Expr:
    return SExpr(op(expr.op), [convert_expr(expr.operand)])


@convert_expr.register
def _(expr: ast.UnaryOp) -> Expr:
    return SExpr(op(expr.op), [convert_expr(expr.operand)])

@convert_expr.register
def _(expr: ast.Tuple) -> Expr:
    if (ctx := convert_expr(expr.ctx)) is not None:
        suffix = f"-{ctx}"
    else:
        suffix = ""
    return SExpr(f"tuple{suffix}", [convert_expr(e) for e in expr.values])

@convert_expr.register
def _(expr: ast.BoolOp) -> Expr:
    return SExpr(op(expr.op), [convert_expr(val) for val in expr.values])