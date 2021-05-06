"""Runtime support for generated parsers"""
import bisect
from dataclasses import dataclass
from typing import Union, TypeVar, Callable, TypeAlias



class Token:
    """A simple token type"""
    __slots__ = "start", "text"
    start: int
    """The starting position (in chars)"""
    text: str
    """The text of the token"""

    def __init__(self, start: int, text: str):
        self.start = start
        self.text = text

    def __eq__(self, other):
        if isinstance(other, str):
            return self.text == other
        elif isinstance(other, Token):
            return type(self) == type(other) \
                and self.text == other.text
        else:
            return NotImplemented

    def __str__(self) -> str:
        return self.text

    @property
    def span(self) -> Span:
        start = self.start
        return Span(start, start + len(self.text))

class Ident(Token):
    pass

@dataclass
class Span:
    __slots__ = "start", "end"
    start: Union[int, Location]
    """The start location"""
    end: Union[int, Location, None]
    """The end location (optional)

    If this is none, then this span only points at a single location"""

    @property
    def resolved(self) -> bool:
        """If this span has been "resolved" to contain
        location objects under the hood (instead of raw bytes)"""
        return isinstance(self.start, Location) and (
            self.end is None or isinstance(self.end, Location))

@dataclass(frozen=True)
class Location:
    line: int
    char_offset: int

class LocationTracker:
    text: str
    _line_starts: list[int]

    def __init__(self, text: str):
        self.text = text
        self._line_starts = [0]
        index = 0
        while (index := text.find('\n', index)) >= 0:
            self._line_starts.append(index + 1)

    def resolve_span(self, target: Span) -> Span:
        """Resolve the specified span in-place"""
        if isinstance(target.start, Location):
            pass
        else:
            target.start = self.resolve_span(target.start)
        if target.end is None or isinstance(target.end, Location):
            pass
        else:
            target.end = self.resolve_span(target.end)
        return target

    def resolve_location(self, char_index: int) -> Location:
        assert isinstance(char_index, int)
        assert char_index >= 0
        line = bisect.bisect_right(self._line_starts, char_index)
        # This returns the element to the right of the line start
        # so bisect_right([0, 4, 8], 0) -> 1
        assert line > 0
        line_start = self._line_starts[line - 1]
        assert char_index >= line_start
        return Location(
            line=line,
            char_offset=char_index-line_start
        )

T = TypeVar("T")
ParseFunc = Union[str, type[Token], Callable[Parser, T]]

class ParseError(BaseException):
    span: Span
    def __init__(self, span: Span, msg: str):
        super().__init__(msg)
        self.span = span

class UnexpectedTokenError(ParseError):
    pass

class UnexpectedEnd(ParseError):
    pass

T = TypeVar("T")
PredictionTable: TypeAlias = dict[Union[str, type], Callable[Parser, T]]

class Parser:
    __slots__ = "tokens", "_current_index"
    tokens: list[Token]
    _current_index: int

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self._current_index = 0

    def peek(self) -> Optional[Token]:
        try:
            return self.tokens[self._current_index]
        except IndexError:
            return None

    def predict(self, table: PredictionTable) -> T:
        token = self.peek()
        if token is None:
            return None
        val = table.get(token.text)
        if val is not None:
            return val
        return table.get(type(token))

    def parse_repeated(
        self, parse_func: ParseFunc,
        minimum: int = 0,
        seperator: Optional[str] = None,
        allow_extra_terminator: bool = False
    ) -> list:
        assert minimum in (0, 1)
        result = []
        if seperator is None:
            while True:
                try:
                    result.append(self.expect(parse_func))
                except UnexpectedTokenError as cause:
                    if len(result) < minimum:
                        # Insufficient items
                        #   -> propagate cause
                        raise
                    else:
                        break
        else:
            had_seperator = False
            while True:
                try:
                    result.append(self.expect(parse_func))
                except UnexpectedTokenError:
                    if had_seperator or len(result) < minimum:
                        # Either insufficient items,
                        # or we had a seperator before.
                        # Either way we expected an element.
                        raise
                    else:
                        # Ignore. There are clearly no more
                        # elements
                        break
                if self.peek() == seperator:
                    self._current_index += 1
                    had_seperator = True
                else:
                    break
            if allow_extra_terminator:
                if self.peek() == seperator:
                    self._current_index += 1

        return result

    def expect(self, parse_func: ParseFunc):
        if isinstance(parse_func, str):
            if self.peek() == parse_func:
                self._current_index += 1
            else:
                raise self.unexpected_token()
        elif isinstance(parse_func, type):
            token = self.peek()
            if isinstance(token, parse_func):
                return token
            else:
                raise UnexpectedTokenError(
                    token.span,
                    f"Expected a {parse_func!r}: {token!r}"
                )
        elif callable(parse_func):
            # We finally deserve have our namesake
            return parse_func(self)
        else:
            raise TypeError(f"Unexpected parse_func type: {type(parse_func)!r}")
    def unexpected_token(self) -> ParseError:
        if (token := self.peek()) is not None:
            raise UnexpectedTokenError(token.span, f"Unexpected token {token.text!r}")
        else:
            raise UnexpectedTokenError(Span(
                start=self.tokens[-1].span.end,
            ), "Unexpected EOF")

