from contextlib import contextmanager
import dataclasses
from .lexer import LiteralString
from pathos.runtime import Ident

class Printer:
    include_newline: bool
    indent_level: int
    def __init__(self, indent_level: int = 0):
        self.include_newline = True
        self.indent_level = indent_level

    def print_indentation(self):
        print("  " * self.indent_level, sep='', end='')

    def pretty_print_text(self, name, value):
        if self.include_newline:
            self.print_indentation()
        if name is not None:
            print(name, '=', sep='', end='')
        if isinstance(value, str):
            print(value, end='')
        else:
            self.pretty_print(value, name=None)
        if self.include_newline:
            print()
    def pretty_print(self, value, *, name: str):
        global indent
        if value is None:
            self.pretty_print_text(name, "None")
        elif isinstance(value, (str, int, float, Ident, LiteralString)):
            self.pretty_print_text(name, repr(value))
        elif name is not None:
            self.pretty_print_text(name, value)
        elif isinstance(value, (list, tuple)):
            opening, closing = ('[', ']') if isinstance(value, list) \
                else ('(', ')')
            if self.include_newline:
                self.print_indentation()
            print(opening, end='')
            single_line = len(value) in (0, 1)
            if single_line:
                indentation_amount = 0
            else:
                indentation_amount = 1
                print()
            self.indent_level += indentation_amount
            try:
                for item in value:
                    if not single_line:
                        self.print_indentation()
                    with self.set_include_newline(False):
                        self.pretty_print(item, name=None)
                    if not single_line:
                        print(',')
            finally:
                self.indent_level -= indentation_amount
            if not single_line:
                self.print_indentation()
            print(closing, end='')
            if self.include_newline:
                print()
        elif dataclasses.is_dataclass(value):
            fields = dataclasses.fields(value)
            print(type(value).__name__ + "(")
            with self.indent():
                for field in fields:
                    self.print_indentation()
                    with self.set_include_newline(False):
                        self.pretty_print(
                            name=field.name,
                            value=getattr(value, field.name),
                        )
                    print(",")
            self.print_indentation()
            print(')', end='')
            if self.include_newline:
                print()
        else:
            raise TypeError(f"Unable to print {type(value)!r}")


    @contextmanager
    def indent(self):
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1

    @contextmanager
    def set_include_newline(self, value: bool):
        old = self.include_newline
        try:
            self.include_newline = value
            yield
        finally:
            self.include_newline = old