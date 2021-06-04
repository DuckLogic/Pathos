#! /usr/bin/env python
"""Generate C code from an ASDL description."""

import os
import sys
import textwrap
import re
import dataclasses

from abc import ABCMeta, abstractmethod
from typing import Union, Optional
from argparse import ArgumentParser
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from .. import asdl

TABSIZE = 4
MAX_COL = 100
AUTOGEN_MESSAGE = "// File automatically generated by {}.\n\n"

RESERVED_NAMES = {
    "type": "ast_type"
}
BUILTIN_TYPE_MAP = {
    "identifier": "Ident",
    "string": "String",
    "int": "i32",
    "constant": "Constant"
}

class NameStyle(Enum):
    SNAKE_CASE = "snake_case"
    UPPER_SNAKE_CASE = "UPPER_SNAKE_CASE"
    PASCAL_CASE = "PascalCase"

    def convert(self, other: str):
        if '_' in other:
            parts = other.split('_')
        else:
            pending = []
            parts = []
            idx = 0
            length = len(other)
            def flush_pending():
                assert pending
                parts.append(''.join(pending))
                pending.clear()
            while idx < length:
                c = other[idx]
                if c.isupper():
                    if pending and pending[-1].islower():
                        flush_pending()
                    pending.append(c)
                else:
                    pending.append(c)
                idx += 1
            if pending:
                flush_pending()
        return self.convert_parts(parts)

    def convert_parts(self, parts: list[str]) -> str:
        if self == NameStyle.SNAKE_CASE:
            return '_'.join(part.lower() for part in parts)
        elif self == NameStyle.UPPER_SNAKE_CASE:
            return '_'.join(part.upper() for part in parts)
        elif self == NameStyle.PASCAL_CASE:
            return ''.join(part[0].upper() + part[1:].lower() for part in parts)
        else:
            raise AssertionError(repr(self))


def rust_type(name):
    """Return a string for the Rust name of the type.

    This function special cases the default types provided by asdl.
    """
    if name in BUILTIN_TYPE_MAP:
        return BUILTIN_TYPE_MAP[name]
    else:
        return NameStyle.PASCAL_CASE.convert(name)

def reflow_lines(s, depth):
    """Reflow the line s indented depth tabs.

    Return a sequence of lines where no line extends beyond MAX_COL
    when properly indented.  The first line is properly indented based
    exclusively on depth * TABSIZE.  All following lines -- these are
    the reflowed lines generated by this function -- start at the same
    column as the first character beyond the opening { in the first
    line.
    """
    size = MAX_COL - depth * TABSIZE
    if len(s) < size:
        return [s]

    lines = []
    cur = s
    padding = ""
    WHITESPACE_PATTERN = re.compile(r"\s")
    while len(cur) > size:
        i = -1
        for val in WHITESPACE_PATTERN.finditer(s, 0, size):
            i = val.pos
        # XXX this should be fixed for real
        if i == -1 and 'GeneratorExp' in cur:
            i = size + 3
        assert i != -1, "Impossible line %d to reflow: %r" % (size, s)
        lines.append(padding + cur[:i])
        if len(lines) == 1:
            # find new size based on brace
            j = cur.find('{', 0, i)
            if j >= 0:
                j += 2 # account for the brace and the space after it
                size -= j
                padding = " " * j
            else:
                j = cur.find('(', 0, i)
                if j >= 0:
                    j += 1 # account for the paren (no space after it)
                    size -= j
                    padding = " " * j
        cur = cur[i+1:]
    else:
        lines.append(padding + cur)
    return lines

def reflow_rust_string(s, depth):
    return '"%s"' % s.replace('\n', '\\n"\n%s"' % (' ' * depth * TABSIZE))

def is_simple(sum):
    """Return True if a sum is a simple.

    A sum is simple if its types have no fields, e.g.
    unaryop = Invert | Not | UAdd | USub
    """
    for t in sum.types:
        if t.fields:
            return False
    return True

def asdl_of(name, obj):
    if isinstance(obj, asdl.Product) or isinstance(obj, asdl.Constructor):
        fields = ", ".join(map(str, obj.fields))
        if fields:
            fields = "({})".format(fields)
        return "{}{}".format(name, fields)
    else:
        if is_simple(obj):
            types = " | ".join(type.name for type in obj.types)
        else:
            sep = "\n{}| ".format(" " * (len(name) + 1))
            types = sep.join(
                asdl_of(type.name, type) for type in obj.types
            )
        return "{} = {}".format(name, types)

class EmitVisitor(asdl.VisitorBase):
    """Visit that emits lines"""

    def __init__(self, file):
        self.file = file
        self.identifiers = set()
        self.singletons = set()
        self.types = set()
        self.depth = 0
        super(EmitVisitor, self).__init__()

    @contextmanager
    def indent(self, amount: int = 1):
        self.depth += amount
        try:
            yield
        finally:
            self.depth -= amount

    def emit_identifier(self, name):
        self.identifiers.add(str(name))

    def emit_singleton(self, name):
        self.singletons.add(str(name))

    def emit_type(self, name):
        self.types.add(str(name))

    def emit(self, s, reflow=True):
        # XXX reflow long lines?
        if reflow:
            lines = reflow_lines(s, self.depth)
        else:
            lines = [s]
        for line in lines:
            if line:
                line = (" " * TABSIZE * self.depth) + line
            self.file.write(line + "\n")

@dataclass
class SharedAttribute:
    name: str
    rust_type: str
    doc: str = "Shared attribute"

SPAN_ATTRS = {"lineno", "col_offset", "end_lineno", "end_col_offset"}
class RustVisitor(EmitVisitor, metaclass=ABCMeta):
    def rewrite_attributes(self, attrs: list[asdl.Field]) -> list[SharedAttribute]:
        attr_map = {attr.name: attr for attr in attrs}
        res = {}
        if attr_map.keys() >= SPAN_ATTRS:
            assert "span" not in attr_map, attr_map["span"]
            for name in SPAN_ATTRS:
                actual = attr_map[name]
                assert actual.type == "int", \
                    f"Unexpected attr {actual!r}"
                del attr_map[name]
            res["span"] = SharedAttribute(
                name="span",
                rust_type="Span",
                doc="The span of the source"
            )
        for val in attr_map.values():
            assert not val.seq and not val.opt, repr(val)
            res[val.name] = SharedAttribute(
                name=val.name,
                rust_type=rust_type(val.type)
            )
        return list(res.values())

    def visitModule(self, mod):
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, tp):
        self.visit(tp.value, tp.name)

    def visitSum(self, sum, name):
        if is_simple(sum):
            self.simple_sum(sum, name)
        else:
            self.sum_with_constructors(sum, name)

@dataclass
class VisitorArg:
    name: str
    rust_type: str

    def __str__(self) -> str:
        return f"{self.name}: {self.rust_type}"

@dataclass
class VisitorMethod:
    name: str
    args: list[VisitorArg]
    return_type: str
    body: Optional[str] = None

class AbstractVisitorGenerator(RustVisitor, metaclass=ABCMeta):
    @abstractmethod
    def visitModule(self, mod):
        pass

    @abstractmethod
    def arg_type(self, parent: asdl.AST, field: Union[asdl.Field, SharedAttribute]) -> str:
        pass

    @abstractmethod
    def product_return_type(self, item: asdl.Product) -> str:
        pass

    @abstractmethod
    def cons_return_type(self, item: asdl.Constructor) -> str:
        pass

    @abstractmethod
    def enum_return_type(
        self, item: asdl.Sum,
    ) -> str:
        pass

    def simple_sum(self, sum, name):
        # We don't do anything for simple sums
        # Visiting a c-style enum is stupid
        pass 

    def emit_visitor_method(self, method: VisitorMethod):
        self.emit(f"fn visit_{method.name}(")
        with self.indent(2):
            self.emit("&mut self,")
            for arg in method.args:
                self.emit(f"{arg},")
        closing = '{' if method.body is not None else ';'
        self.emit(f") -> {method.return_type} {closing}")
        if method.body is not None:
            with self.indent():
                for line in method.body:
                    self.emit(line)
            self.emit("}")

    def sum_variant_name(self, sum: asdl.Sum, tp: asdl.Constructor):
        sum_name = NameStyle.SNAKE_CASE.convert(sum.name)
        tp_name = NameStyle.SNAKE_CASE.convert(tp.name)
        return f"{sum_name}_{tp_name}"

    def sum_with_constructors(self, sum, name):
        def emit(s):
            self.emit(s)
        assert sum.name == name
        for tp in sum.types:
            assert isinstance(tp, asdl.Constructor), f"Bad type: {tp!r}"
            method = self.sum_variant_visitor(sum, tp)
            if method is not None:
                self.emit_visitor_method(method)

    def sum_variant_visitor(
        self,
        parent: asdl.Sum,
        variant: asdl.Constructor,
    ) -> Optional[VisitorMethod]:
        return_type = self.enum_return_type(parent, skip_duplicates=True)
        args=[]
        for field in self.rewrite_attributes(parent.attributes):
            arg_type = self.arg_type(variant, field)
            args.append(VisitorArg(name=field.name, rust_type=arg_type))
        for field in variant.fields:
            args.append(VisitorArg(
                name=field.name,
                rust_type=self.arg_type(variant, field)
            ))
        return VisitorMethod(
            name=self.sum_variant_name(parent, variant),
            args=args, return_type=return_type
        )

    def visitConstructor(self, cons, name):
        visitor = self.constructor_visitor(cons, name)
        if visitor is not None:
            self.emit_visitor_method(visitor)

    def visitProduct(self, prod, name):
        visitor = self.product_visitor(prod, name)
        if visitor is not None:
            self.emit_visitor_method(visitor)


    def constructor_visitor(self, cons: asdl.Constructor) -> Optional[VisitorMethod]:
        return VisitorMethod(
            name=NameStyle.SNAKE_CASE.convert(cons.name),
            args=[f"{field.name}: {self.arg_type(cons, field)}"
                for field in cons.fields],
            return_type=self.cons_return_type(cons)
        )

    def visitField(self, field):
        raise NotImplementedError

    def product_visitor(self, product: asdl.Product, name: str):
        args = []
        for field in self.rewrite_attributes(product.attributes):
            # rudimentary attribute handling
            arg_type = self.arg_type(product, field)
            args.append(VisitorArg(
                name=field.name,
                rust_type=arg_type
            ))
        for field in product.fields:
            arg_type = self.arg_type(product, field)
            args.append(VisitorArg(
                name=field.name,
                rust_type=arg_type,
            ))
        return VisitorMethod(
            name=NameStyle.SNAKE_CASE.convert(name),
            args=args, return_type=self.product_return_type(product)
        )

@dataclass
class AssociatedVisitorType:
    name: str
    bound: str = None
    doc: str = None
    impl: str = None

    def __str__(self):
        res = [f"Self::{self.name}"]
        if self.bound is not None:
            res.append(f": {self.bound}")
        return ''.join(res)

    def write_decl(self, out: EmitVisitor):
        if self.doc is not None:
            out.emit(f"/// {self.doc}")
        res = [f"type {self.name}"]
        if self.impl is not None:
            res.append(f" = {self.impl}")
        elif self.bound is not None:
            res.append(f": {self.bound}")
        res.append(";")
        out.emit(''.join(res))

class TypeInfo:
    simple_types: set[str]
    type_info: dict[str, asdl.Type]
    def __init__(self):
        self.simple_types = set()
        self.type_info = {}

    def is_simple(self, name: str) -> bool:
        return name in BUILTIN_TYPE_MAP \
            or name in self.simple_types

class ClarrifyingVisitor(asdl.VisitorBase):
    info: TypeInfo
    def __init__(self, info: TypeInfo):
        super().__init__()
        assert isinstance(info, TypeInfo)
        self.info = info

    def visitModule(self, mod):
        for d in mod.dfns:
            self.visit(d)

    def visitType(self, tp):
        self.info.type_info[tp.name] = tp
        self.visit(tp.value, tp.name)

    def visitSum(self, sum, name):
        sum.name = name
        if is_simple(sum):
            self.info.simple_types.add(name)
        for tp in sum.types:
            assert isinstance(tp, asdl.Constructor)
            # Give access to parent
            setattr(tp, 'parent_sum', sum)
            self.visit(tp, tp.name)

    def visitField(self, field):
        if field.name in RESERVED_NAMES:
            field.name = RESERVED_NAMES[field.name]

    def visitProduct(self, product, name):
        product.name = name
        for f in product.fields:
            self.visitField(f)

    def visitConstructor(self, cons, name):
        cons.name = name
        for f in cons.fields:
            self.visitField(f)

class GenericVisitorGenerator(AbstractVisitorGenerator):
    """Generate a generic visitor,
    completely generic over result types (via associated types).

    This is how we avoid constructing an AST."""
    pending_associated_types: dict[str, AssociatedVisitorType]
    info: TypeInfo
    def __init__(self, info: TypeInfo, file):
        super().__init__(file)
        assert isinstance(info, TypeInfo)
        self.info = info
        self.pending_associated_types = dict()

    def visitModule(self, mod, emit_trait=True):
        if emit_trait:
            self.emit("pub trait AstVisitor {")
        with self.indent(1 if emit_trait else 0):
            for dfn in mod.dfns:
                self.visit(dfn)
            self.emit("")
            for associated in self.pending_associated_types.values():
                associated.write_decl(self)
            self.pending_associated_types.clear()
        if emit_trait:
            self.emit("}")

    def arg_type(self, parent: asdl.AST, field: Union[asdl.Field, SharedAttribute]) -> str:
        if isinstance(field, SharedAttribute):
            return field.rust_type
        if field.type in BUILTIN_TYPE_MAP:
            type_name = BUILTIN_TYPE_MAP[field.type]
        elif self.info.is_simple(field.type):
            type_name = rust_type(field.type)
        else:
            type_name = f"Self::{rust_type(field.type)}"
        if field.opt:
            assert not field.seq
            return f"Option<{type_name}>"
        elif field.seq:
            return f"impl Iterator<Item={type_name}>"
        else:
            return type_name

    def return_type(self, assoc: AssociatedVisitorType, *, skip_duplicates: bool=False) -> str:
        if assoc.name in self.pending_associated_types:
            if skip_duplicates:
               pass
            else:
                raise ValueError(f"Duplicate {assoc.name} -> {assoc!r}") 
        else:
            self.pending_associated_types[assoc.name] = assoc
        return f"Self::{assoc.name}"

    def product_return_type(self, item: asdl.Product) -> AssociatedVisitorType:
        return self.return_type(self.product_assoc_type(item))

    def cons_return_type(self, item: asdl.Constructor) -> str:
        return self.return_type(self.cons_assoc_type(item))

    def enum_return_type(
        self, item: asdl.Sum,
        **kwargs
    ) -> AssociatedVisitorType:
        return self.return_type(self.enum_assoc_type(item), **kwargs)


    def product_assoc_type(self, item: asdl.Product) -> AssociatedVisitorType:
        return AssociatedVisitorType(rust_type(item.name))

    def cons_assoc_type(self, item: asdl.Constructor) -> str:
        return AssociatedVisitorType(rust_type(item.name))

    def enum_assoc_type(
        self, parent: asdl.Sum,
    ) -> AssociatedVisitorType:
        return AssociatedVisitorType(rust_type(parent.name))


ASSOCIATED_TYPE_PATTERN = re.compile(r"Self::(\w+)")
OPTIONAL_ASSOCIATED_TYPE_PATTERN = re.compile(r"Option<Self::(\w+)>")
class MemoryVisitorGenerator(GenericVisitorGenerator):

    def build_struct(self, name: str, args: list[VisitorArg]):
        res = []
        res.append(f"{name} {{")
        small_fields = []
        big_fields = []
        for arg in args:
            if (match := ASSOCIATED_TYPE_PATTERN.match(arg.rust_type)) \
                and not self.info.is_simple(match[1]):
                big_fields.append(f"{arg.name}: Box::new({arg.name})")
            elif (match := OPTIONAL_ASSOCIATED_TYPE_PATTERN.match(arg.rust_type)) \
                and not self.info.is_simple(match[1]):
                big_fields.append(f"{arg.name}: {arg.name}.map(Box::new)")
            elif "impl Iterator" in arg.rust_type:
                big_fields.append(f"{arg.name}: {arg.name}.collect::<Vec<_>>()")
            else:
                small_fields.append(f"{arg.name}")
        with self.indent():
            tab = ' ' * TABSIZE
            if small_fields:
                for line in textwrap.wrap(
                    ', '.join(sorted(small_fields)) + ",",
                    width=MAX_COL-(self.depth * TABSIZE)
                ):
                    res.append(tab + line)
            for field in sorted(big_fields):
                res.append(f"{tab}{field},")
        res.append("}")
        return res

    def sum_variant_visitor(
        self,
        parent: asdl.Sum,
        variant: asdl.Constructor,
    ) -> Optional[VisitorMethod]:
        visitor = super().sum_variant_visitor(parent, variant)
        parent_name = NameStyle.PASCAL_CASE.convert(parent.name)
        variant_name = NameStyle.PASCAL_CASE.convert(variant.name)
        visitor.body = self.build_struct(f"{parent_name}::{variant_name}", visitor.args)
        return visitor

    def constructor_visitor(self, cons: asdl.Constructor):
        visitor = super().constructor_visitor(cons)
        cons_name = NameStyle.PASCAL_CASE.convert(cons.name)
        visitor.body = self.build_struct(f"{parent_name}::{variant_name}", visitor.args)
        return visitor

    def product_visitor(self, product: asdl.Product, name: str):
        visitor = super().product_visitor(product, name)
        cons_name = NameStyle.PASCAL_CASE.convert(name)
        visitor.body = self.build_struct(f"{cons_name}", visitor.args)
        return visitor

    def product_assoc_type(self, item: asdl.Product) -> AssociatedVisitorType:
        assoc = super().product_assoc_type(item)
        assoc.doc = f"The concrete implementation of {item.name}"
        assoc.impl = rust_type(item.name)
        return assoc

    def cons_assoc_type(self, item: asdl.Constructor) -> str:
        assoc = super().cons_assoc_type(item)
        assoc.doc = f"The concrete implementation of {item.name}"
        assoc.impl = rust_type(item.name)
        return assoc

    def enum_assoc_type(
        self, parent: asdl.Sum,
    ) -> AssociatedVisitorType:
        base_name = rust_type(parent.name)
        assoc = super().enum_assoc_type(parent)
        assoc.doc = f"The concrete implementation of enum {base_name}"
        assoc.impl = base_name
        return assoc


    def visitModule(self, mod):
        self.emit("pub struct MemoryVisitor;")
        self.emit("impl AstVisitor for MemoryVisitor {")
        with self.indent():
            super().visitModule(mod, emit_trait=False)
        self.emit("}")

class RustTypeDeclareVisitor(RustVisitor):
    info: TypeInfo
    def __init__(self, info: TypeInfo, *args):
        super().__init__(*args)
        self.info = info
        self.inside_enum = False

    @contextmanager
    def switch_inside_enum(self, /, new_val: bool):
        old_val = self.inside_enum
        self.inside_enum = new_val
        yield
        self.inside_enum = old_val

    def visitModule(self, mod):
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, tp):
        self.visit(tp.value, tp.name)

    def visitSum(self, sum, name):
        if is_simple(sum):
            self.simple_sum(sum, name)
        else:
            self.sum_with_constructors(sum, name)

    def simple_sum(self, sum, name):
        enum = []
        for i, tp in enumerate(sum.types):
            enum.append(f"{tp.name}={i + 1},")
        type_name = rust_type(name)
        self.emit("#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]")
        self.emit(f"pub enum {type_name} {{")
        with self.indent():
            for line in enum:
                self.emit(line)
        self.emit("}")
        self.emit("")

    def sum_with_constructors(self, sum, name):
        def emit(s):
            self.emit(s)
        emit("#[derive(Educe, Debug, Clone)]")
        emit("#[educe(PartialEq, Eq, Hash)]")
        emit(f"pub enum {rust_type(name)} {{")
        with self.indent():
            for idx, tp in enumerate(sum.types):
                emit(f"{tp.name} {{")
                with self.indent():
                    for field in self.rewrite_attributes(sum.attributes):
                        # rudimentary attribute handling
                        attr_type = field.rust_type
                        if field.doc is not None:
                            emit(f"/// {field.doc}")
                        self.emit('#[educe(Hash(ignore))]')
                        self.emit('#[educe(PartialEq(ignore))]')
                        emit(f"{field.name}: {attr_type},")
                    with self.switch_inside_enum(True):
                        self.visit(tp)
                emit("},")
        emit("}")
        def emit_match(field, *, borrowed=True):
            ownership_pat = "ref " if borrowed else ""
            emit("match *self {")
            for idx, tp in enumerate(sum.types):
                with self.indent():
                    last = f"=> {field.name}," if idx == len(sum.types) - 1 else "|"
                    emit(f"{rust_type(name)}::{tp.name} {{ {ownership_pat}{field.name}, .. }} {last}")
            emit("}")
        self.emit_rewritten_attrs(
            sum.attributes,
            name=name,
            handler=emit_match
        )

    def emit_rewritten_attrs(self, attrs, *, name, handler):
        def emit(*args):
            self.emit(*args)
        def emit_handler(field: SharedAttribute, *, pub=True, borrowed=True):
            attr_type = str(field.rust_type)
            if field.doc is not None:
                emit(f"/// {field.doc}")
            result_type = rust_type(attr_type)
            if borrowed:
                result_type = "&" + result_type
            vis = "pub " if pub else ""
            emit(f"{vis}fn {field.name}(&self) -> {result_type} {{")
            with self.indent():
                handler(field, borrowed=borrowed)
            emit("}")
        if attrs:
            rewritten_attrs=self.rewrite_attributes(attrs)
            span_field=None
            for field in list(rewritten_attrs):
                if field.name == "span" and field.rust_type == "Span":
                    span_field=field
                    rewritten_attrs.remove(field)
            if rewritten_attrs:
                emit(f"impl {rust_type(name)} {{")
                with self.indent():
                    for field in rewritten_attrs:
                        emit_handler(field)
                emit("}")
            if span_field is not None:
                emit(f"impl Spanned for {rust_type(name)} {{")
                with self.indent():
                    emit_handler(dataclasses.replace(
                        span_field, doc=None
                    ), pub=False, borrowed=False)
                emit("}")


    def visitConstructor(self, cons):
        if cons.fields:
            amount = 0
            if not self.inside_enum:
                amount = 1
                self.emit("#[derive(Educe, Debug, Clone)]")
                self.emit("#[educe(PartialEq, Eq, Hash)]")
                self.emit(f"pub struct {cons.name} {{")
            with self.indent(amount):
                for f in cons.fields:
                    self.visit(f)
            if not self.inside_enum:
                self.emit("}")
            self.emit("")

    def visitField(self, field):
        # XXX need to lookup field.type, because it might be something
        # like a builtin...
        type_name = rust_type(field.type)
        if not self.info.is_simple(field.type) \
            and not field.seq:
            type_name = f"Box<{type_name}>"
        name = field.name
        vis = "pub " if not self.inside_enum else ""
        if field.opt:
            assert not field.seq, repr(field)
            type_name = f"Option<{type_name}>"
        elif field.seq:
            type_name = f"Vec<{type_name}>"
        self.emit(f"{vis}{name}: {type_name},")

    def visitProduct(self, product, name):
        self.emit("#[derive(Educe, Debug, Clone)]")
        self.emit("#[educe(PartialEq, Eq, Hash)]")
        self.emit(f"pub struct {rust_type(name)} {{")
        with self.indent():
            for f in product.fields:
                self.visit(f)
            for field in self.rewrite_attributes(product.attributes):
                # rudimentary attribute handling
                self.emit('#[educe(Hash(ignore))]')
                self.emit('#[educe(PartialEq(ignore))]')
                self.emit(f"pub {field.name}: {field.rust_type},");
        self.emit("}")
        def emit_field_access(field, *, borrowed=True):
            ownership = "&" if borrowed else ""
            self.emit(f"{ownership}self.{field.name}")
        self.emit_rewritten_attrs(
            product.attributes,
            name=name,
            handler=emit_field_access
        )
        self.emit("")

def ast_func_name(name):
    return f"_PyAST_{name}"



class ChainOfVisitors:
    def __init__(self, *visitors):
        self.visitors = visitors

    def visit(self, obj):
        for v in self.visitors:
            v.visit(obj)
            if isinstance(v, EmitVisitor):
                v.emit("")


def write_source(mod, f):
    info = TypeInfo()
    v = ChainOfVisitors(
        ClarrifyingVisitor(info=info),
        RustTypeDeclareVisitor(info, f),
        GenericVisitorGenerator(info, f),
        MemoryVisitorGenerator(info, f),
    )
    v.visit(mod)

def main(input_filename, rust_filename, dump_module=False):
    auto_gen_msg = AUTOGEN_MESSAGE.format("/".join(Path(__file__).parts[-2:]))
    mod = asdl.parse(input_filename)
    if dump_module:
        print('Parsed Module:')
        try:
            from prettyprinter import register_pretty, \
                install_extras, \
                pprint as pretty_print
        except ImportError:
            print("WARN: Failed to import 'prettyprinter'", file=sys.stderr)
            pretty_print = print
        else:
            install_extras()
        pretty_print(mod)
    if not asdl.check(mod):
        sys.exit(1)

    with rust_filename.open("w") as rust_file:
        rust_file.write(auto_gen_msg)

        write_source(mod, rust_file)

    print(f"{rust_filename}, regenerated.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("-R", "--rust-file", type=Path, required=True)
    parser.add_argument("-d", "--dump-module", action="store_true")

    args = parser.parse_args()
    main(args.input_file, args.rust_file, args.dump_module)

