import ast
from typing import Callable
from pathlib import Path

import click
import sys
from click import ClickException

from pathos import ast_utils

@click.group()
def pathos():
    """A set of utilities related to the Pathos parser"""
    pass


@pathos.command('dump-node')
@click.option(
    '--mode', type=click.Choice(["eval", "exec", "single", "func_type"]),
    help="The compilation mode given to ast.parse"
)
@click.option('--text', 'explicit_text', help="Pass the input source as a CLI option")
@click.option('--file', type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Read source from the specified file")
@click.option('--stdin', 'should_read_stdin', is_flag=True, help="Read sources from stdin")
@click.option('--edit', 'open_editor', is_flag=True, help="Open an editor (like vim) to create the input")
@click.option('--indent-size', help="The size of indentation used for pretty printing", type=int, default=4)
@click.option('--source-filename', help="Specify the filename of the source file")
def dump_node(
        explicit_text: str, file: Path, should_read_stdin: bool,
        open_editor, mode: str,
        indent_size: int, source_filename: str
):
    """Pretty print an AST node"""
    specified_input_sources: list[tuple[str, str, str, Callable[[], str]]] = []
    if explicit_text is not None:
        specified_input_sources.append(("Explicit text input", "explicit_text.py", "eval", lambda: explicit_text))
    if file is not None:
        def read_file():
            with file.open('rt', encoding='utf8') as f:
                return f.read()
        specified_input_sources.append((f"File {str(file)!r}", file.name, "exec", read_file))
    if should_read_stdin:
        def read_stdin():
            return click.prompt("Please enter the input source below:\n")
        specified_input_sources.append(("Standard input", "inputted.py", "eval", read_stdin))
    if open_editor:
        def open_edit():
            res = click.edit(extension='.py')
            if res is None:
                raise click.ClickException("Nothing returned from the editor!")
            return res
        specified_input_sources.append(("An editor", "edited.py", "exec", open_edit))
    if not specified_input_sources:
        raise click.ClickException("No input source specified")
    elif len(specified_input_sources) > 1:
        raise click.ClickException("Multiple input sources specified: " + ', '.join(name for name, func in specified_input_sources))
    else:
        _, fallback_source_name, default_mode, func = specified_input_sources[0]
        if source_filename is None:
            source_filename = fallback_source_name
        if mode is None:
            mode = default_mode
        text = func()
        assert isinstance(text, str), repr(text)
    try:
        tree = ast.parse(source=text, filename=source_filename,mode=mode)
    except SyntaxError as e:
        raise click.ClickException(f"SyntaxError: {e}")
    dumper = ast_utils.PrettyDumper(indent_size=indent_size)
    dumper.pretty_dump(tree, newline=True)
    for line in str(dumper).splitlines():
        print(line)

@pathos.command('sexpr-ast')
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