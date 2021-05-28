from abc import ABCMeta, abstractmethod
from contextlib import contextmanager, AbstractContextManager

class CodeWriter(metaclass=ABCMeta):
    indent_size: int
    def __init__(self, *, indent_size=4):
        self._lines = []
        self._pending_line = []
        self._current_indent = 0
        self.indent_size = indent_size
    
    @contextmanager
    def indent(self, *, amount: int = 1):
        assert amount >= 1
        try:
            self._current_indent += amount
            yield
        finally:
            self._current_indent -= amount

    def print(self, *items, sep: str=',', end: str='\n'):
        line = (sep or '').join(map(str, items))
        self._pending_line.append(line)
        if end == '\n':
            indent = ' ' * self.indent_size * self._current_indent
            pending = ''.join(self._pending_line)
            self._pending_line.clear()
            if '\n' in pending:
                idx = 0
                while idx >= 0:
                    next_idx = pending.find('\n', idx + 1)
                    end = next_idx if next_idx >= 0 else len(pending)
                    self.print(pending[idx:end], end='\n')
                    idx = next_idx
            if not pending:
                self.lines.append(pending)
            joined = indent + pending
            self.lines.append(joined)
        elif end:
            assert '\n' not in end
            self._pending_line.append(end)


CodeWriterCtx = AbstractContextManager[CodeWriter]

@dataclass
class Arg:
    name: str
    static_type: Optional[str] = None

class CodeGenerator(metaclass=ABCMeta):
    writer: CodeWriter
    def __init__(self, writer: CodeWriter):
        assert isinstance(writer, CodeWriter)
        self.writer = writer

    @abstractmethod
    def declare_method(
        self, name: str, *,
        args: list[Arg],
        return_type: str
    ) -> CodeWriterCtx:
        pass

