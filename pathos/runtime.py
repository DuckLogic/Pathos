"""Runtime support for generated parsers"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Union, TypeVar, Callable


class Token:
    """A simple token type"""
    __slots__ = "span", "text"
    span: Span
    """The starting position (in chars)"""
    text: str
    """The text of the token"""

    def __init__(self, span: Span, text: str):
        self.span = span
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

    def __repr__(self) -> str:
        return f"{self.text!r} @ {self.span}"

class Ident(Token):
    pass

@dataclass
class Span:
    __slots__ = "start", "end"
    start: Location
    """The start location"""
    end: Optional[Location]
    """The end location (optional)

    If this is none, then this span only points at a single location"""

    @property
    def resolved(self) -> bool:
        """If this span has been "resolved" to contain
        location objects under the hood (instead of raw bytes)"""
        return isinstance(self.start, Location) and (
            self.end is None or isinstance(self.end, Location))

    def __repr__(self) -> str:
        start, end = self.start, self.end
        if end is not None and start != end:
            if start.line != end.line:
                return f"{start}..{end}"
            else:
                return f"{start.line}:{start.char_offset}..{end.char_offset}"
        else:
            return repr(start)


@dataclass(frozen=True)
class Location:
    line: int
    char_offset: int

    def __repr__(self):
        return f"{self.line}:{self.char_offset}"

class LocationTracker:
    text: str
    _line_starts: list[int]

    def __init__(self, text: str):
        self.text = text
        self._line_starts = [0]
        index = 0
        while (index := text.find('\n', index)) >= 0:
            self._line_starts.append(index + 1)
            index += 1

    def create_span(self, start: int, end: Optional[int]) -> Span:
        """Create a resolved span"""
        return Span(
            start=self.resolve_location(start),
            end=self.resolve_location(end) \
                if end is not None else None,
        )

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

class ParseError(BaseException):
    span: Span
    def __init__(self, span: Span, msg: str):
        super().__init__(f"{msg} @ {span}")
        self.span = span


class UnexpectedTokenError(ParseError):
    pass

class UnexpectedEnd(ParseError):
    pass

class Parser:
    __slots__ = "tokens", "_current_index"
    tokens: list[Token]
    _current_index: int

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self._current_index = 0

    def peek(self, *, ahead: int = 0) -> Optional[Token]:
        assert ahead >= 0
        try:
            return self.tokens[self._current_index + ahead]
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
        if isinstance(parse_func, (str, Token)):
            if self.peek() == parse_func:
                self._current_index += 1
                return parse_func # String
            else:
                raise self.unexpected_token()
        elif isinstance(parse_func, type):
            token = self.peek()
            if isinstance(token, parse_func):
                self._current_index += 1
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


T = TypeVar("T")
ParseFunc = Union[str, type[Token], Callable[Parser, T]]
