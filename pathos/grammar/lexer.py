"""Lexer for the grammar"""

from pathos.runtime import Token, LocationTracker, Span, ParseError

class Keyword(Token):
    pass

class LiteralString(Token):
    value: str
    def __init__(self, start: int, text: str, value: str):
        super().__init__(start=start, text=text)
        self.value = value

SPECIAL_TOKENS = frozenset((
    "{", "}", "!", "@", "$", "*", "**", "+", "++",
    "=", "(", ")", "[", "]", "~", "\"", "'", ":", ";",
    "|", ","
))
IDENTIFIER_START_CHARS = frozenset([
    *map(chr, range(ord('a'), ord('z')+1)),
    *map(chr, range(ord('A'), ord('Z')+1)),
    '_', "$"
])
IDENTIFIER_CONTINUE_CHARS = frozenset((
    *IDENTIFIER_START_CHARS,
    *map(chr, range(ord('0', ord('9')+1)))
))

class LexError(ParseError):
    pass

class Lexer:
    __slots__ = "text", "current_index", "keywords", \
        "special_tokens"
    text: str
    current_index: int
    keywords: frozenset[str]
    special_tokens: frozenset[str]

    def __init__(self, text: str, keywords: frozenset[str], special_tokens: frozenset[str]):
        keywords = frozenset(keywords)
        special_tokens = frozenset(special_tokens)
        # Validate our assumption about multi-char 'special tokens'
        # That each multi-char special token is made up entirely
        # of special-char prefixes
        for special in self.special_tokens:
            if special == 0:
                raise ValueError("Empty special token")
            elif len(special) == 1:
                continue
            for index in range(0, len(special) - 1):
                if (prefix := special[:index]) not in special_tokens:
                    raise ValueError(f"Special token {special!r} contains non-special prefix {prefix!r}")
        # Set fields
        self.text = text
        self.current_index = 0
        self.keywords = keywords
        self.special_tokens = special_tokens

    def parse_next(self) -> Optional[Token]:
        start = self.current_index
        text = self.text
        length = len(text)
        while c := text[start].isblank():
            start += 1
            if start > length:
                return None
        special_tokens = self.special_tokens
        c = text[start]
        if c in IDENTIFIER_START_CHARS:
            ident = self.parse_ident()
            if ident.text in self.keywords:
                return Keyword(ident.start, ident.text)
            else:
                return ident
        elif c == '"':
            index = start
            chars = []
            while index < length:
                c = text[index]
                if c == '"':
                    self.current_index = index + 1
                    return LiteralString(
                        start=start,
                        value=''.join(chars),
                        text=text[start:index+1]
                    )
                elif c == '\\':
                    try:
                        escape = text[index + 1]
                    except IndexError:
                        raise ParseError(
                            Span(start=index),
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
                            Span(start=index, end=index+1),
                            f"Invalid escape char '{escape}'"
                        )
                    index += 1
                    continue
                else:
                    chars.append(c)
                index += 1
            raise ParseError(
                Span(start=start, end=index),
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
            return Token(start=start, text=special_text)
        else:
            raise ParseError(
                Span(start=start_index, end=start_index+1),
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
        return Ident(start, self.text[start:index])

def tokenize(text: str):
    index = 0
    text_length = len(text)
    def advance_while(, test: )
    while index < text_length:
        c = text[index]
        if c in IDENTIFIER_START_CHARS:
            start = index
            index += 1
            while index < text_length and (c := ):

        result = 
