//! A lexer for python-style source code
use std::hash::{Hash, Hasher};
use std::borrow::Borrow;
use std::cell::RefCell;
use bumpalo::Bump;

use either::Either;

#[cfg(feature="num-bigint")]
/// An arbitrary precision integer
pub type BigInt = num_bigint::BigInt;
#[cfg(fefature="rug")]
/// A rug BigInt
pub type BigInt = rug::Integer;
#[cfg(not(any(feature="num-bigint", feature="rug")))]
/// How integers are stored if big integers are disabled
pub struct BigInt {
    text: String
}

use hashbrown::HashMap;
use logos::{Logos, Lexer};

/// A python identifier
///
/// These are interned, so there should
/// never be any duplicates within the same
/// source file.
#[derive(Clone)]
pub struct Ident<'a> {
    text: &'a str,
    hash: u64,
    /// A monotonically increasing id
    /// for this identifier
    id: u32
}
impl Borrow<str> for Ident<'_> {
    #[inline]
    fn borrow(&self) -> &str {
        self.text
    }
}
impl Hash for Ident<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}
impl Eq for Ident<'_> {}
impl PartialEq for Ident<'_> {
    #[inline]
    fn eq(&self, other: &Ident) -> bool {
        std::ptr::eq(self.text, other.text)
    }
}
impl std::fmt::Debug for Ident<'_> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{:?}", self.text)
    }
}
struct RawLexerState {
    starting_line: bool,
}
impl Default for RawLexerState {
    #[inline]
    fn default() -> Self {
        RawLexerState {
            // We start out beginning a line
            starting_line: true,
        }
    }
}

#[derive(Debug)]
pub enum LexError {
    Unknown
}

pub struct PythonLexer<'src, 'arena: 'src> {
    arena: &'arena Bump,
    text: &'src str,
    raw_lexer: Lexer<'src, RawToken<'src>>,
    known_idents: hashbrown::HashMap<&'arena Ident<'arena>, ()>,
    pending_indentation_change: isize,
    indent_stack: Vec<usize>
}

macro_rules! translate_tokens {
    ($target:expr; $($name:ident $( ( $mtch:ident ) )? $(=> $handler:expr)?),*) => {{
        use self::RawToken::*;
        match $target {
            $(RawToken::$name $( ( $mtch ) )? => {
                translate_tokens!(handler for $name $( ($mtch) )? $(=> $handler)?)
            }),*
        }
    }};
    (handler for $name:ident $(( $mtch:ident ) )? => $handler:expr) => ($handler);
    (handler for $name:ident) => (Token::$name);
}
impl<'src, 'arena> PythonLexer<'src, 'arena> {
    pub fn new(arena: &'arena Bump, text: &'src str) -> Self {
        PythonLexer {
            arena, text,
            raw_lexer: RawToken::lexer(text),
            known_idents: ::hashbrown::HashMap::default(),
            pending_indentation_change: 0,
            indent_stack: vec![0]
        }
    }
    pub fn reset(&mut self, text: &'src str) {
        self.text = text;
        self.raw_lexer = RawToken::lexer(text);
        self.pending_indentation_change = 0;
        self.indent_stack.clear();
        self.indent_stack.push(0);
    }
    pub fn lex_all(&mut self, text: &'src str) -> Result<Vec<Token<'arena>>, LexError> {
        self.reset(text);
        let mut res = Vec::new();
        while let Some(token) = self.next()? {
            res.push(token);
        }
        Ok(res)
    } 
    pub fn create_ident(&mut self, text: &'src str) -> &'arena Ident<'arena> {
        use std::hash::BuildHasher;
        use std::convert::TryFrom;
        let old_len = u32::try_from(self.known_idents.len())
            .expect("Too many identifiers");
        let mut hasher = self.known_idents.hasher().build_hasher();
        hasher.write(text.as_bytes());
        let hash = hasher.finish();
        match self.known_idents.raw_entry_mut().from_hash(hash, |other| {
            other.hash == hash && other.text == text
        }) {
            hashbrown::hash_map::RawEntryMut::Occupied(entry) => {
                *entry.key()
            },
            hashbrown::hash_map::RawEntryMut::Vacant(entry) => {
                let allocated_text = self.arena.alloc_str(text);
                let key = self.arena.alloc(Ident {
                    text: allocated_text, hash, id: old_len
                });
                entry.insert_hashed_nocheck(hash, key, ());
                assert!(self.known_idents.len() > old_len as usize);
                key
            }
        }
    }
    #[allow(unused)]
    pub fn next(&mut self) -> Result<Option<Token<'arena>>, LexError> {
        if self.pending_indentation_change != 0 {
            if self.pending_indentation_change < 0 {
                self.pending_indentation_change += 1;
                return Ok(Some(Token::DecreaseIndent));
            } else if self.pending_indentation_change > 0 {
                self.pending_indentation_change -= 1;
                return Ok(Some(Token::IncreaseIndent));
            } else { unreachable!() }
        }
        'yum: loop {
            // Lets do some post processing ;)
            let token = match self.raw_lexer.next() {
                Some(val) => val,
                None => return Ok(None) // EOF
            };
            return Ok(Some(translate_tokens!(token;
                /*
                 * The 'raw' lexer filters out all spaces and tabs
                 * unless it is at the start of a line.
                 * However, it doesn't do the implicit comparison
                 * against the stack as described here:
                 * https://docs.python.org/3.9/reference/lexical_analysis.html#indentation
                 */
                RawIndent (amount) => {
                    let current_top = *self.indent_stack.last().unwrap();
                    if amount == current_top {
                        continue 'yum;
                    } else if amount > current_top {
                        self.indent_stack.push(amount);
                        return Ok(Some(Token::IncreaseIndent));
                    } else {
                        assert_eq!(self.pending_indentation_change, 0);
                        while amount < *self.indent_stack.last().unwrap() {
                            self.indent_stack.pop();
                            self.pending_indentation_change -= 1;
                            assert!(!self.indent_stack.is_empty());
                        }
                        assert!(self.pending_indentation_change < 0);
                        self.pending_indentation_change += 1;
                        assert!(!self.indent_stack.is_empty());
                        return Ok(Some(Token::DecreaseIndent));
                    }
                },
                RawNewline => Token::Newline, // TODO: Do we need to do anything else?
                Integer (inner) => {
                    match inner {
                        Either::Left(int) => Token::IntegerLiteral(int),
                        Either::Right(int) => Token::BigIntegerLiteral(self.arena.alloc(int))
                    }
                },
                Identifier (text) => {
                    Token::Ident(self.create_ident(text))
                },
                FloatLiteral (f) => Token::FloatLiteral(f),
                Error => {
                    return Err(LexError::Unknown)
                },
                String(s) => {
                    Token::StringLiteral(self.arena.alloc(StringInfo {
                        prefix: s.prefix,
                        raw: s.raw,
                        quote_style: s.quote_style,
                        original_text: self.arena.alloc_str(s.original_text)
                    }))
                },
                // keywords
                False, Await, Else, Import, Pass, None,
                True, Class, Finally, Is, Return, And,
                Continue, For, Lambda, Try, As, Def, From,
                Nonlocal, While, Assert, Del, Global, Not,
                With, Async, Elif, If, Or, Yield,
                // operators
                Plus, Minus, Star, DoubleStar, Slash, DoubleSlash,
                Percent, At, LeftShift, RightShift, Ampersand,
                BitwiseOr, BitwiseXor, BitwiseInvert, AssignOperator,
                LessThan, GreaterThan, LessThanOrEqual, GreaterThanOrEqual,
                DoubleEquals, NotEquals, OpenParen, CloseParen,
                OpenBracket, CloseBracket, OpenBrace, CloseBrace,
                Comma, Colon, Period, Semicolon, Equals, Arrow,
                PlusEquals, MinusEquals, StarEquals, SlashEquals,
                DoubleSlashEquals, PercentEquals, AtEquals,
                AndEquals, OrEquals, XorEquals, RightShiftEquals,
                LeftShiftEquals, DoubleStarEquals
            )));
        }
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'arena> {
    // **************
    //    Keywords
    // **************
    False, // 0
    Await, // 1
    Else, // 2
    Import, // 3
    Pass, // 4
    None, // 5
    True, // 6
    Class, // 7
    Finally, // 8
    Is, // 9
    Return, // 10
    And, // 11
    Continue, // 12
    For, // 13
    Lambda, // 14
    Try, // 15
    As, // 16
    Def, // 17
    From, // 18
    Nonlocal, // 19
    While, // 20
    Assert, // 21
    Del, // 22
    Global, // 23
    Not, // 24
    With, // 25
    Async, // 26
    Elif, // 27
    If, // 28
    Or, // 29
    Yield, // 30
    // ***************
    //    Operators
    // ***************
    Plus, // 1
    Minus, // 2
    Star, // 3
    DoubleStar, // 4
    Slash, // 5
    DoubleSlash, // 6
    Percent, // 7
    At, // 8
    LeftShift, // 9
    RightShift, // 10
    Ampersand, // 11
    BitwiseOr, // 12
    BitwiseXor, // 13
    BitwiseInvert, // 14
    AssignOperator, // 15
    LessThan, // 16
    GreaterThan, // 17
    LessThanOrEqual, // 18
    GreaterThanOrEqual, // 19
    DoubleEquals, // 20
    NotEquals, // 21
    OpenParen, // 22
    CloseParen, // 23
    OpenBracket, // 24
    CloseBracket, // 25
    OpenBrace, // 26
    CloseBrace, // 27
    Comma, // 28
    Colon, // 29
    Period, // 30
    Semicolon, // 31
    Equals, // 32
    Arrow, // 33
    PlusEquals, // 34
    MinusEquals, // 35
    StarEquals, // 36
    SlashEquals, // 37
    DoubleSlashEquals, // 39
    PercentEquals, // 38
    AtEquals, // 40
    AndEquals, // 41
    OrEquals, // 42
    XorEquals, // 43
    RightShiftEquals, // 44
    LeftShiftEquals, // 45
    DoubleStarEquals, // 46
    IntegerLiteral(i64),
    FloatLiteral(f64),
    /// An arbitrary-precision integer,
    /// that is too big to fit in a regular int.
    BigIntegerLiteral(&'arena BigInt),
    /// A string literal
    // TODO: Actually parse the underlying text
    StringLiteral(&'arena StringInfo<'arena>),
    /// An identifier.
    ///
    /// Do not confuse this with [Token::IncreaseIndent].
    /// This has the shorter name because it is more common.
    Ident(&'arena Ident<'arena>),
    /// Increase the indentation
    IncreaseIndent,
    /// Decrease the indentation
    DecreaseIndent,
    /// A logical newline
    Newline,
}

#[derive(Logos, Debug, PartialEq)]
#[logos(extras = RawLexerState)]
enum RawToken<'src> {
    // ********I******
    //    Keywords
    // **************
    #[token("False")]
    False, // 0
    #[token("await")]
    Await, // 1
    #[token("else")]
    Else, // 2
    #[token("import")]
    Import, // 3
    #[token("pass")]
    Pass, // 4
    #[token("None")]
    None, // 5
    #[token("True")]
    True, // 6
    #[token("class")]
    Class, // 7
    #[token("finally")]
    Finally, // 8
    #[token("is")]
    Is, // 9
    #[token("return")]
    Return, // 10
    #[token("and")]
    And, // 11
    #[token("continue")]
    Continue, // 12
    #[token("for")]
    For, // 13
    #[token("lambda")]
    Lambda, // 14
    #[token("try")]
    Try, // 15
    #[token("as")]
    As, // 16
    #[token("def")]
    Def, // 17
    #[token("from")]
    From, // 18
    #[token("nonlocal")]
    Nonlocal, // 19
    #[token("while")]
    While, // 20
    #[token("assert")]
    Assert, // 21
    #[token("del")]
    Del, // 22
    #[token("global")]
    Global, // 23
    #[token("not")]
    Not, // 24
    #[token("with")]
    With, // 25
    #[token("async")]
    Async, // 26
    #[token("elif")]
    Elif, // 27
    #[token("if")]
    If, // 28
    #[token("or")]
    Or, // 29
    #[token("yield")]
    Yield, // 30
    // ***************
    //    Operators
    // ***************
    #[token("+")]
    Plus, // 1
    #[token("-")]
    Minus, // 2
    #[token("*")]
    Star, // 3
    #[token("**")]
    DoubleStar, // 4
    #[token("/")]
    Slash, // 5
    #[token("//")]
    DoubleSlash, // 6
    #[token("%")]
    Percent, // 7
    #[token("@")]
    At, // 8
    #[token("<<")]
    LeftShift, // 9
    #[token(">>")]
    RightShift, // 10
    #[token("&")]
    Ampersand, // 11
    #[token("|")]
    BitwiseOr, // 12
    #[token("^")]
    BitwiseXor, // 13
    #[token("~")]
    BitwiseInvert, // 14
    #[token(":=")]
    AssignOperator, // 15
    #[token("<")]
    LessThan, // 16
    #[token(">")]
    GreaterThan, // 17
    #[token("<=")]
    LessThanOrEqual, // 18
    #[token(">=")]
    GreaterThanOrEqual, // 19
    #[token("==")]
    DoubleEquals, // 20
    #[token("!=")]
    NotEquals, // 21
    #[token("(")]
    OpenParen, // 22
    #[token(")")]
    CloseParen, // 23
    #[token("[")]
    OpenBracket, // 24
    #[token("]")]
    CloseBracket, // 25
    #[token("{")]
    OpenBrace, // 26
    #[token("}")]
    CloseBrace, // 27
    #[token(",")]
    Comma, // 28
    #[token(":")]
    Colon, // 29
    #[token(".")]
    Period, // 30
    #[token(";")]
    Semicolon, // 31
    #[token("=")]
    Equals, // 32
    #[token("->")]
    Arrow, // 33
    #[token("+=")]
    PlusEquals, // 34
    #[token("-=")]
    MinusEquals, // 35
    #[token("*=")]
    StarEquals, // 36
    #[token("/=")]
    SlashEquals, // 37
    #[token("//=")]
    DoubleSlashEquals, // 39
    #[token("%=")]
    PercentEquals, // 38
    #[token("@=")]
    AtEquals, // 40
    #[token("&=")]
    AndEquals, // 41
    #[token("|=")]
    OrEquals, // 42
    #[token("^=")]
    XorEquals, // 43
    #[token(">>=")]
    RightShiftEquals, // 44
    #[token("<<=")]
    LeftShiftEquals, // 45
    #[token("**=")]
    DoubleStarEquals, // 46
    // ********************
    //    Special Tokens
    // ********************
    /// A python identifier
    #[regex(r"[\p{XID_Start}_][\p{XID_Continue}_]*", ident)]
    Identifier(&'src str),
    #[regex(r"[1-9]([_1-9])*|0([_0])*", parse_decimal_int)]
    #[regex(r"0[xX][_0-9A-F]+", parse_hex_int)]
    #[regex(r"0[bB][_01]+", parse_bin_int)]
    #[regex(r"0[oO][_0-9]+", parse_octal_int)]
    Integer(Either<i64, BigInt>),
    /// A python string literal
    ///
    /// No backslashes or escapes have been interpreted
    /// in any way. It's up to the parser to do that.
    ///
    /// TODO: We should interpret slashes ourselves.
    /// 
    /// We should also use a sub-parser, just like shown here:
    /// https://github.com/maciejhirsz/logos/blob/99d8f4ce/tests/tests/lexer_modes.rs#L11
    #[regex(r##"([rRuUfFbB]|[Ff][rR]|[rR][fFBb]|[Bb][rR])?(["']|"""|''')"##, lex_string)]
    String(StringInfo<'src>),
    #[regex(r##"(([0-9][_0-9]+)?\.([0-9][_0-9]+)|([0-9][_0-9]+)\.)([eE][+-]?([0-9][_0-9]+))?"##, lex_float)]
    #[regex(r##"(([0-9][_0-9]+)?(\.)?([0-9][_0-9]+)?)([eE][+-]?([0-9][_0-9]+))"##, lex_float)]
    FloatLiteral(f64),
    #[error]
    #[token(r#"#"#, skip_comment)]
    Error,
    /// A raw newline
    ///
    /// NOTE: This is implicitly skipped if there is a backslash
    /// right before it.
    #[regex(r"(\n|\r\n)", newline)]
    #[regex(r"\\(\n|\r\n)", logos::skip)]
    RawNewline,
    /// Raw indentation at the start of a newline
    ///
    /// The value of the variant indicates the amount of
    /// indentation (both tabs and spaces).
    #[regex(r"[\t ]+", raw_indentation)]
    RawIndent(usize),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StringPrefix {
    /// A format string prefix.
    Formatted,
    /// An explicit unicode prefix.
    ///
    /// Note that this is redundant on Python 3. 
    Unicode,
    /// A byte string prefix
    Bytes
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QuoteStyle {
    Double,
    Single,
    DoubleLong,
    SingleLong,
}
impl QuoteStyle {
    #[inline]
    pub fn start_byte(self) -> u8 {
        match self {
            QuoteStyle::Single |
            QuoteStyle::SingleLong => b'\'',
            QuoteStyle::Double |
            QuoteStyle::DoubleLong => b'"',
        }
    }
    #[inline]
    pub fn text(self) -> &'static str {
        match self {
            QuoteStyle::DoubleLong => r#"""""#,
            QuoteStyle::SingleLong => r"'''",
            QuoteStyle::Double => r#"""#,
            QuoteStyle::Single => r"'",
        }
    }
    #[inline]
    pub fn is_triple_string(self) -> bool {
        match self {
            QuoteStyle::Single |
            QuoteStyle::Double => false,
            QuoteStyle::SingleLong |
            QuoteStyle::DoubleLong => true,
        }
    }
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct StringInfo<'src> {
    /// An explicit string prefix,
    /// or none if it's just a plain string
    pub prefix: Option<StringPrefix>,
    /// True if the string is also a raw string
    pub raw: bool,
    pub quote_style: QuoteStyle,
    /// The original text of this string.
    ///
    /// Backslashes have not been interpreted
    pub original_text: &'src str
}
fn skip_comment<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> logos::Skip {
    debug_assert_eq!(lex.slice(), "#");
    if let Some(line_end) = memchr::memchr(b'\n', lex.remainder().as_bytes()) {
        lex.bump(line_end + 1);
    } else {
        // No newline -> Everything till EOF is a comment
        lex.bump(lex.remainder().len());
    }
    logos::Skip
}
fn lex_float<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> f64 {
    // This should not fail
    ::lexical_core::parse_format::<f64>(
        lex.slice().as_bytes(),
        ::lexical_core::NumberFormat::PYTHON3_LITERAL,
    ).unwrap()
}
fn raw_indentation<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> logos::Filter<usize> {
    if !lex.extras.starting_line {
        return logos::Filter::Skip;
    }
    let mut count = 0;
    for b in lex.slice().bytes() {
        match b {
            b' ' => {
                count += 1;
            },
            b'\t' => {
                count += count % 8;
            }
            _ => unreachable!("Unexpected byte: {:?}", b),
        }
    }
    assert!(count > 0);
    logos::Filter::Emit(count)
}
fn newline<'a>(lex: &mut Lexer<'a, RawToken<'a>>) {
    lex.extras.starting_line = true;
}
#[inline]
fn ident<'src>(lex: &mut Lexer<'src, RawToken<'src>>) -> &'src str {
    lex.slice() // We don't intern till later ;)
}
#[cold]
fn fallback_parse_int<'arena>(
    radix: i64, text: &str
) -> BigInt {
    /*
     * We get here if the regular integer parse code overflows.
     *
     * We fallback to parsing as a arbitrary precision integer.
     */
    #[cfg(feature="num-bigint")]
    {
        assert!(radix >= 1);
        let mut result = BigInt::default();
        for b in text.bytes() {
            let digit_val = match b {
                b'0'..=b'9' => b - b'0',
                b'A'..=b'F' => b - b'A',
                b'a'..=b'f' => b - b'a',
                b'_' => continue, // Ignore underscores
                _ => unreachable!("Invalid byte: {:?}", b)
            };
            result *= radix;
            result += digit_val as i64;
        }
        result
    }
    #[cfg(feature="rug")]
    {
        let parsed = BigInt::parse_radix(text, radix).unwrap();
        BigInt::from(parsed)
    }
    #[cfg(not(any(feature="num-bigint", feature="rug")))]
    {
        BigInt {
            text: String::from(text)
        }
    }
}
macro_rules! int_parse_func {
    ($name:ident, radix = $radix:literal, strip = |$s:ident| $strip_code:expr) => {
        fn $name<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> Either<i64, BigInt> {
            // Eagerly attempt to parse as an `i64`
            let mut result = 0i64;
            let remaining: &str = {
                let $s = lex.slice();
                $strip_code
            };
            const RADIX: i64 = $radix;
            for b in remaining.bytes() {
                let digit_value = match b {
                    b'0'..=b'9' => b - b'0',
                    b'A'..=b'F' => b - b'A',
                    b'a'..=b'f' => b - b'a',
                    b'_' => continue,
                    _ => unreachable!("Forbidden digit: {:?}", b)
                } as i64;
                result = match result.checked_mul(RADIX)
                    .and_then(|res| res.checked_add(digit_value)) {
                    Some(result) => result,
                    None => {
                        return Either::Right(fallback_parse_int(
                            RADIX, remaining
                        ));
                    }
                };
            }
            Either::Left(result)
        }
    }
}
int_parse_func!(
    parse_decimal_int,
    radix = 10,
    strip = |s| s
);
int_parse_func!(
    parse_hex_int,
    radix = 16,
    strip = |s| {
        debug_assert_eq!(&s[0..1], "0");
        debug_assert!(matches!(&s[1..2], "x" | "X"));
        s
    }
);

int_parse_func!(
    parse_bin_int,
    radix = 2,
    strip = |s| {
        debug_assert_eq!(&s[0..1], "0");
        debug_assert!(matches!(&s[1..2], "b" | "B"));
        &s[2..]
    }
);

int_parse_func!(
    parse_octal_int,
    radix = 8,
    strip = |s| {
        debug_assert_eq!(&s[0..1], "0");
        debug_assert!(matches!(&s[1..2], "o" | "O"));
        &s[2..]
    }
);
fn lex_string<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> Result<StringInfo<'a>, StringError> {
    /*
     * Already parsed:
     * 1. Optionally: A prefix
     * 2. A string start, either triple string or single string
     */
    let mut info = {
        let slice = lex.slice();
        /*
         * NOTE: The lexer isn't affected by string prefixes.
         * That's interpreted by a later stage in the parser.
         *
         * We only care about the particular suffix, whether it's
         * long string, or short string and whether it uses tripple quotes or single quotes.
         */
        let mut index = 1;
        // NOTE: Known ahead of time to be ASCII
        let (prefix, raw) = match &slice[..1] {
            "r" | "R" => {
                // Valid next chars: [fFbB]?
                let prefix = match &slice[1..2] {
                    "f" | "F" => {
                        index = 2; // consumed
                        Some(StringPrefix::Formatted)
                    },
                    "b" | "B" => {
                        index = 2; // Consumed
                        Some(StringPrefix::Bytes)
                    },
                    _ => None
                };
                (prefix, true)
            },
            "u" | "U" => {
                /*
                 * NOTE: No other prefix can come after
                 * a 'u' starting token
                 */
                (Some(StringPrefix::Unicode), false)
            },
            "f" | "F" => {
                // Valid next chars: [rR]?
                let raw = match &slice[1..2] {
                    "r" | "R" => {
                        index = 2; // consumed
                        true
                    },
                    _ => false
                };
                (Some(StringPrefix::Formatted), raw)
            },
            "b" | "B" => {
                // Valid next chars: [rR]?
                let raw = match &slice[1..2] {
                    "r" | "R" => {
                        index = 2; // consumed
                        true
                    },
                    _ => false
                };
                (Some(StringPrefix::Bytes), raw)
            },
            c => {
                if cfg!(debug_assertions) {
                    match c {
                        r#"""# | "'" => {},
                        // NOTE: logos should've guarded against this....
                        _ => unreachable!("Invalid prefix: {}", c)
                    }
                }
                index = 0;
                (None, false)
            }
        };
        // Match on the remaining characters
        let remaining = &slice[index..];
        let quote_style = match remaining {
            r#"""# => QuoteStyle::Double,
            r#"'"# => QuoteStyle::Single,
            r#"""""# => QuoteStyle::DoubleLong,
            r#"'''"# => QuoteStyle::SingleLong,
            _ => unreachable!("Unexpected chars")
        };
        StringInfo { prefix, raw, quote_style, original_text: "" }
    };
    /*
     * NOTE: This starts at the end of the
     * current token. Thus we can essume the next
     * (non-escaped) instance of `quote_style` closes
     * the string literal. The only escape character
     * that we need to worry about is '\'' and '\"'.
     * for single and double quotes respectively.
     */
    let remaining_bytes = lex.remainder().as_bytes();
    let expected_closing_text = info.quote_style.text();
    for index in ::memchr::memchr2_iter(info.quote_style.start_byte(), b'\n', remaining_bytes) {
        /*
         * NOTE: One of the coolest properties of UTF8 is
         * that if a valid UTF8 substring exists an index `i`,
         * it can be assumed to be a character boundary.
         *
         * In other words, there is no chance we will run
         * across ACII '\' or '"' characters unless they
         * are actually valid characters in the UTF8 string :0.
         * The coolest thing is that this works for BMP and non-BMP
         * characters too. It's all automatic.
         *
         * Therefore, all we have to do to confirm this potential
         * match is make sure that this character isn't escaped
         * (by checking bytes[index-1] != '\') and then double-check
         * to make sure we have the proper clsoing
         * in the case that we're a triple string.
         */
        if remaining_bytes.get(index - 1) == Some(&b'\\') {
            continue; // Skip escaped quote (or newline)
        }
        if remaining_bytes[index] == b'\n' {
            if info.quote_style.is_triple_string() {
                continue; // just ignore the newline
            } else {
                // newline is an error...
                return Err(StringError::ForbiddenNewline);
            }
        }
        let end = if info.quote_style.is_triple_string() {
            match (info.quote_style, remaining_bytes.get(index..index + 3)) {
                (QuoteStyle::DoubleLong, Some(br#"""""#)) |
                (QuoteStyle::SingleLong, Some(br"'''")) => {
                    // We found our closing bytes.
                    index + 3
                },
                (QuoteStyle::DoubleLong, _) |
                (QuoteStyle::SingleLong, _) => continue,
                (QuoteStyle::Double, _) |
                (QuoteStyle::Single, _) => unreachable!()
            }
        } else {
            index + 1
        };
        debug_assert_eq!(&remaining_bytes[index..end], expected_closing_text.as_bytes());
        lex.bump(end);
        info.original_text = lex.slice();
        return Ok(info);
    }
    Err(StringError::NoClosingQuote) // no closing paren
}
#[derive(Debug)]
enum StringError {
    NoClosingQuote,
    ForbiddenNewline
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn basic() {
        let arena = Bump::new();
        let mut lexer = PythonLexer::new(&arena, "");
        assert_eq!(
            lexer.lex_all("def foo(a, b, cat)").unwrap(),
            vec![
                Token::Def,
                Token::Ident(lexer.create_ident("foo")),
                Token::OpenParen,
                Token::Ident(lexer.create_ident("a")),
                Token::Comma,
                Token::Ident(lexer.create_ident("b")),
                Token::Comma,
                Token::Ident(lexer.create_ident("cat")),
                Token::CloseParen
            ]
        );
    }
}
