from .. import CodeWriter, CodeGenerator, Arg
from contextlib import contextmanager

class PythonCodeWriter(CodeWriter):
    pass

class PythonCodeGenerator(CodeGenerator):
    def __init__(self, writer: PythonCodeWriter):
        assert isinstance(writer, PythonCodeWriter)
        super().__init__(writer=writer)

    @contextmanager
    def declare_method(
        self, name: str, *,
        args: list[Arg],
        return_type: str
    ) -> CodeWriterCtx:
        def format_arg(arg: Arg):
            if arg.static_type is not None:
                return f"{arg.name}: {arg.static_type}"
            else:
                return arg.name
        self.writer.print(
            f"def {name}({', '.join(map(format_arg, args))})",
            f" -> {return_type}:" if return_type is not None else ":"
        )
        with self.writer.indent():
            yield self.writer
