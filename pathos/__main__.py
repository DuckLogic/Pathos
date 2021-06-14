import ast

import click
import sys

from pathos import ast_utils

@click.group()
def pathos():
    """A set of utilities related to the Pathos parser"""
    pass

@pathos.command('cpython-ast')
@click.option('--expr', required=True, help="The input expression")
def cpython_ast(expr: str):
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        print(f"Invalid syntax: {e}", file=sys.stderr)
        sys.exit(1)
    pretty = ast_utils.PrettyPrinter()
    ast_utils.convert_expr(tree).pretty_print(pretty)
    print("Pretty AST:")
    for line in str(pretty).splitlines():
        print(line)



if __name__ == "__main__":
    pathos()