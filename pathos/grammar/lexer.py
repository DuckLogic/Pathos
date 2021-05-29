"""Lexer for the grammar"""
from typing import Optional

from pathos.runtime import Token, LocationTracker, \
    Ident, Span, ParseError

class Keyword(Token):
    pass

class LiteralString(Token):
    value: str
    def __init__(self, span: Span, text: str, value: str):
        super().__init__(span=span, text=text)
        self.value = value

SPECIAL_TOKENS = frozenset((
    "{", "}", "!", "@", "$", "*", "**", "+", "++",
    "=", "(", ")", "[", "]", "~", ":", ";", '&',
    "|", ",", '-', '<', '>', '?', '^', '.', "->"
))
IDENTIFIER_START_CHARS = frozenset([
    *map(chr, range(ord('a'), ord('z')+1)),
    *map(chr, range(ord('A'), ord('Z')+1)),
    '_', "$"
])
IDENTIFIER_CONTINUE_CHARS = frozenset((
    *IDENTIFIER_START_CHARS,
    *map(chr, range(ord('0'), ord('9')+1))
))

def is_valid_ident(s: str) -> bool:
    if not s:
        return False
    valid_start, valid_continue = IDENTIFIER_START_CHARS, \
        IDENTIFIER_CONTINUE_CHARS
    return s[0] in valid_start and \
        all(map(valid_continue.__contains__, s[1:]))


class LexError(ParseError):
    pass

class Lexer:
    __slots__ = "text", "current_index", "keywords", \
        "special_tokens", "tracker"
    text: str
    current_index: int
    keywords: frozenset[str]
    special_tokens: frozenset[str]
    tracker: LocationTracker

    def __init__(self, text: str, keywords: frozenset[str], special_tokens: frozenset[str]):
        keywords = frozenset(keywords)
        special_tokens = frozenset(special_tokens)
        # Validate our assumption about multi-char 'special tokens'
        # That each multi-char special token is made up entirely
        # of special-char prefixes
        for special in special_tokens:
            if special == 0:
                raise ValueError("Empty special token")
            elif len(special) == 1:
                continue
            for index in range(1, len(special) - 1):
                if (prefix := special[:index]) not in special_tokens:
                    raise ValueError(f"Special token {special!r} contains non-special prefix {prefix!r}")
        # Set fields
        self.tracker = LocationTracker(text=text)
        self.text = text
        self.current_index = 0
        self.keywords = keywords
        self.special_tokens = special_tokens

    def _skip_whitespace(self) -> Optional[bool]:
        """Attempt to skip whitespace, returning None on EOF"""
        start = self.current_index
        text = self.text
        index = start
        try:
            while text[index].isspace():
                index += 1
        except IndexError:
            return None # EOF
        self.current_index = index
        return index > start

    def _skip_comment(self) -> bool:
        """Attempt to skip comments"""
        start = self.current_index
        text = self.text
        index = start
        if text.startswith("/*", index):
            try:
                end = text.find("*/", index)
            except ValueError:
                raise LexError(
                    Span(start, start + 2),
                    "Unable to find end of comment '*/'"
                )
            self.current_index = end + 2
            return True
        elif text.startswith("//", index):
            try:
                end = text.find("\n")
            except ValueError:
                # No end of line, just comment till end
                end = len(text) - 1
            self.current_index = end + 1
            return True
        else:
            return False # No comment found


    def parse_next(self) -> Optional[Token]:
        text = self.text
        length = len(text)
        while True:
            if self._skip_whitespace() is None:
                return None # EOF
            if not self._skip_comment():
                break # Something that isn't a comment
        start = self.current_index
        special_tokens = self.special_tokens
        c = text[start]
        if c in IDENTIFIER_START_CHARS:
            ident = self.parse_ident()
            if ident.text in self.keywords:
                return Keyword(ident.start, ident.text)
            else:
                return ident
        elif c in ('"', '\''):
            open_kind = c
            index = start + 1
            chars = []
            while index < length:
                c = text[index]
                if c == open_kind:
                    self.current_index = index + 1
                    return LiteralString(
                        span=self.tracker.create_span(start, None),
                        value=''.join(chars),
                        text=text[start:index+1]
                    )
                elif c == '\\':
                    try:
                        escape = text[index + 1]
                    except IndexError:
                        raise ParseError(
                            self.tracker.create_span(start=index, end=None),
                            "Expected a string escape",
                        ) from None
                    if escape == '"':
                        chars.append('"')
                    elif escape == '\\':
                        chars.append('\\')
                    elif escape == 'n':
                        chars.append('\n')
                    else:
                        raise ParseError(
                            self.tracker.create_span(start=index, end=index+1),
                            f"Invalid escape char '{escape}'"
                        )
                    index += 2
                    continue
                else:
                    chars.append(c)
                index += 1
            raise ParseError(
                self.tracker.create_span(start=start, end=index),
                "Expected to find ending quote '\"'"
                " for string literal"
            )
        elif c in special_tokens:
            index = start + 1
            special_text = c
            while index < len(text) \
                and (next_char := text[index]) in special_tokens:
                if special_text + next_char not in special_tokens:
                    break
                else:
                    special_text += next_char
                    index += 1
                    continue
            assert special_text == text[start:index]
            self.current_index = index
            return Token(
                span=self.tracker.create_span(start, index),
                text=special_text
            )
        else:
            raise ParseError(
                self.tracker.create_span(start, start+1),
                f"Unexpected character {c!r}"
            )

    def parse_ident(self) -> Ident:
        start = self.current_index
        text = self.text
        length = len(self.text)
        assert text[start] in IDENTIFIER_START_CHARS
        index = start + 1
        continue_set = IDENTIFIER_CONTINUE_CHARS
        while index < length and text[index] in continue_set:
            index += 1
        assert index > start
        self.current_index = index
        return Ident(
            self.tracker.create_span(start, index),
            self.text[start:index]
        )

def tokenize(text: str) -> list[Token]:
    lexer = Lexer(
        text=text,
        keywords={''},
        special_tokens=SPECIAL_TOKENS
    )
    result = []
    while (token := lexer.parse_next()) is not None:
        result.append(token)
    return result