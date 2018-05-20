#!/usr/bin/env python3
"""Generate Rust code from an ASDL file"""
import os, sys

from . import asdl

TABSIZE = 4

BUILTIN_TYPES = {
    'identifier': 'Ident',
    'string': 'String',
    'bytes': 'Vec<u8>',
    'int': 'i32',
    'object': 'PyObject',
    'singleton',
                     'constant'}
}

def get_rust_type(name):
    if name
