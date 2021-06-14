from functools import singledispatch
import ast
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Union


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
class SExpr:
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
def op(op: Union[ast.operator, ast.unaryop, ast.boolop]):
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
def _(expr: ast.BoolOp) -> Expr:
    return SExpr(op(expr.op), [convert_expr(val) for val in expr.values])