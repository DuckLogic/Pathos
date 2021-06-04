//! A lexer for python-style source code
use std::hash::{Hash, Hasher};
use std::borrow::Borrow;
use bumpalo::Bump;

use either::Either;

#[macro_export]
macro_rules! tk {
    ('\n') => (Token::Newline);
    (False) => (Token::False);
    (await) => (Token::Await);
    (else) => (Token::Else);
    (import) => (Token::Import);
    (pass) => (Token::Pass);
    (None) => (Token::None);
    (True) => (Token::True);
    (class) => (Token::Class);
    (finally) => (Token::Finallly);
    (is) => (Token::Is);
    (return) => (Token::Return);
    (and) => (Token::And);
    (continue) => (Token::Continue);
    (for) => (Token::For);
    (lambda) => (Token::Lambda);
    (try) => (Token::Try);
    (as) => (Token::As);
    (def) => (Token::Def);
    (from) => (Token::From);
    (nonlocal) => (Token::Nonlocal);
    (while) => (Token::While);
    (assert) => (Token::Assert);
    (del) => (Token::Del);
    (global) => (Token::Global);
    (not) => (Token::Not);
    (with) => (Token::With);
    (async) => (Token::Async);
    (elif) => (Token::Elif);
    (if) => (Token::If);
    (or) => (Token::Or);
    (yield) => (Token::Yield);
    (break) => (Token::Break);
    (except) => (Token::Except);
    (in) => (Token::In);
    (raise) => (Token::Raise);
    // ***************
    //    Operators
    // ***************
    (+) => (Token::Plus);
    (-) => (Token::Minus);
    (*) => (Token::Star);
    (**) => (Token::DoubleStar);
    (/) => (Token::Slash);
    (/ /) => (Token::DoubleSlash);
    ("//") => (Token::DoubleSlash);
    (%) => (Token::Percent);
    (@) => (Token::At);
    (<<) => (Token::LeftShift);
    (>>) => (Token::RightShift);
    (&) => (Token::Ampersand);
    (|) => (Token::BitwiseOr);
    (^) => (Token::BitwiseXor);
    (~) => (Token::BitwiseInvert);
    (:=) => (Token::AssignOperator);
    (<) => (Token::LessThan);
    (>) => (Token::GreaterThan);
    (<=) => (Token::LessThanOrEqual);
    (>=) => (Token::GreaterThanOrEqual);
    (==) => (Token::DoubleEquals);
    (!=) => (Token::NotEquals);
    ('(') => (Token::OpenParen);
    (')') => (Token::CloseParen);
    ('[') => (Token::OpenBracket);
    (']') => (Token::CloseBracket);
    ('{') => (Token::OpenBrace);
    ('}') => (Token::CloseBrace);
    (,) => (Token::Comma);
    (:) => (Token::Colon);
    (.) => (Token::Period);
    (;) => (Token::Semicolon);
    (=) => (Token::Equals);
    (->) => (Token::Arrow);
    (+=) => (Token::PlusEquals);
    (-=) => (Token::MinusEquals);
    (*=) => (Token::StarEquals);
    (/=) => (Token::SlashEquals);
    (/ /=) => (Token::DoubleSlashEquals);
    ("//=") => (Token::DoubleSlashEquals);
    (%=) => (Token::PercentEquals);
    (@=) => (Token::AtEquals);
    (&=) => (Token::AndEquals);
    (|=) => (Token::OrEquals);
    (^=) => (Token::XorEquals);
    (>>=) => (Token::RightShiftEquals);
    (<<=) => (Token::LeftShiftEquals);
    (**=) => (Token::DoubleStarEquals);
}

use logos::{Logos, Lexer};

use crate::ast::constants::{BigInt, QuoteStyle, StringPrefix, StringStyle};

/// A python identifier
///
/// These are interned, so there should
/// never be any duplicates within the same
/// source file.
#[derive(Clone)]
pub struct Ident<'a> {
    text: &'a str,
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
impl Borrow<str> for &'_ Ident<'_> {
    #[inline]
    fn borrow(&self) -> &str {
        self.text
    }
}
impl Hash for Ident<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.text, state)
    }
}
impl Eq for Ident<'_> {}
impl PartialEq for Ident<'_> {
    #[inline]
    fn eq(&self, other: &Ident) -> bool {
        debug_assert_eq!(
            std::ptr::eq(self.text, other.text),
            self.text == other.text,
            concat!("Pointer equality gave different result than",
            " value equality for {:?} and {:?}, {0:p} and {1:p}"),
            self.text, other.text,
        );
        std::ptr::eq(self.text, other.text)
    }
}
impl std::fmt::Debug for Ident<'_> {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{:?}", self.text)
    }
}
struct RawLexerState {
}
impl Default for RawLexerState {
    #[inline]
    fn default() -> Self {
        RawLexerState {
        }
    }
}

#[derive(Debug)]
pub enum LexError {
    Unknown
}

pub struct PythonLexer<'src, 'arena: 'src> {
    arena: &'arena Bump,
    raw_lexer: Lexer<'src, RawToken<'src>>,
    known_idents: hashbrown::HashMap<&'src str, &'arena Ident<'arena>>,
    pending_indentation_change: isize,
    indent_stack: Vec<usize>,
    starting_line: bool,
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
            arena,
            raw_lexer: RawToken::lexer(text),
            known_idents: ::hashbrown::HashMap::default(),
            pending_indentation_change: 0,
            indent_stack: vec![0],
            starting_line: true
        }
    }
    pub fn reset(&mut self, text: &'src str) {
        self.raw_lexer = RawToken::lexer(text);
        self.pending_indentation_change = 0;
        self.indent_stack.clear();
        self.indent_stack.push(0);
        self.starting_line = true;
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
        use std::convert::TryFrom;
        let old_len = u32::try_from(self.known_idents.len())
            .expect("Too many ids");
        match self.known_idents.entry(text) {
            hashbrown::hash_map::Entry::Occupied(entry) => {
                *entry.get()
            },
            hashbrown::hash_map::Entry::Vacant(entry) => {
                let allocated_text = self.arena.alloc_str(text);
                let allocated = self.arena.alloc(Ident {
                    text: allocated_text, id: old_len
                });
                entry.insert(allocated);
                assert!(self.known_idents.len() > old_len as usize);
                allocated
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
            if self.starting_line {
                self.starting_line = false;
                let amount = match self.raw_lexer.remainder().bytes()
                        .position(|b| b != b' ' && b != b'\t')  {
                    Some(amount) => amount,
                    None => 0
                };
                let actual_byte = self.raw_lexer.remainder().as_bytes().get(amount).cloned();
                self.raw_lexer.bump(amount);
                if actual_byte == Some(b'\n') {
                    // It's an entirely blank line. Skip it
                    self.raw_lexer.bump(1);
                    self.starting_line = true;
                    continue 'yum;
                } else if actual_byte == Some(b'#') {
                    /*
                     * It's a comment. Skip till the line end
                     * and don't count it towards indentation.
                     */
                    let remaining = self.raw_lexer.remainder();
                    let line_end = ::memchr::memchr(b'\n', remaining.as_bytes())
                        .unwrap_or(remaining.len());
                    self.raw_lexer.bump(line_end);
                    self.starting_line = true;
                    continue 'yum;
                }
                let current_top = *self.indent_stack.last().unwrap();
                if amount > current_top {
                    self.indent_stack.push(amount);
                    return Ok(Some(Token::IncreaseIndent));
                } else if amount < current_top {
                    assert_eq!(self.pending_indentation_change, 0);
                    while dbg!(amount) < dbg!(*self.indent_stack.last().unwrap()) {
                        self.indent_stack.pop();
                        self.pending_indentation_change -= 1;
                        assert!(!self.indent_stack.is_empty());
                    }
                    assert!(self.pending_indentation_change < 0);
                    self.pending_indentation_change += 1;
                    assert!(!self.indent_stack.is_empty());
                    return Ok(Some(Token::DecreaseIndent));
                } else {
                    assert_eq!(amount, current_top);
                }
            }
            // Lets do some post processing ;)
            let token = match self.raw_lexer.next() {
                Some(val) => val,
                None => return Ok(None) // EOF
            };
            return Ok(Some(translate_tokens!(token;
                RawNewline => {
                    self.starting_line = true;
                    // TODO: Do we need to do anything else?
                    Token::Newline
                },
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
                        style: s.style,
                        original_text: self.arena.alloc_str(s.original_text)
                    }))
                },
                // keywords
                False, Await, Else, Import, Pass, None,
                True, Class, Finally, Is, Return, And,
                Continue, For, Lambda, Try, As, Def, From,
                Nonlocal, While, Assert, Del, Global, Not,
                With, Async, Elif, If, Or, Yield,
                Break, Except, In, Raise,
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
    Break, // 31
    Except, // 32
    In, // 33
    Raise, // 34
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
    #[token("break")]
    Break, // 31
    #[token("except")]
    Except, // 32
    #[token("in")]
    In, // 33
    #[token("raise")]
    Raise, // 34
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
    #[regex(r"\p{Whitespace}", logos::skip)] // Skip whitespace
    Error,
    /// A raw newline
    ///
    /// NOTE: This is implicitly skipped if there is a backslash
    /// right before it.
    #[regex(r"(\n|\r\n)")]
    #[regex(r"\\(\n|\r\n)", logos::skip)]
    RawNewline,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct StringInfo<'src> {
    pub style: StringStyle,
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
    let style = {
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
        StringStyle { prefix, raw, quote_style, }
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
    let expected_closing_text = style.quote_style.text();
    for index in ::memchr::memchr2_iter(style.quote_style.start_byte(), b'\n', remaining_bytes) {
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
            if style.quote_style.is_triple_string() {
                continue; // just ignore the newline
            } else {
                // newline is an error...
                return Err(StringError::ForbiddenNewline);
            }
        }
        let end = if style.quote_style.is_triple_string() {
            match (style.quote_style, remaining_bytes.get(index..index + 3)) {
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
        return Ok(StringInfo {
            style, original_text: lex.slice(),
        });
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
    #[cfg(test)]
    use pretty_assertions::{assert_eq,};
    macro_rules! munch_token {
        ($lexer:expr, $res:expr; Ident($text:literal), $($remaining:tt)*) => {
            $res.push(Token::Ident($lexer.create_ident($text)));
            munch_token!($lexer, $res; $($remaining)*)
        };
        ($lexer:expr, $res:expr; $name:ident, $($remaining:tt)*) => {
            $res.push(Token::$name);
            munch_token!($lexer, $res; $($remaining)*)
        };
        ($lexer:expr, $res:expr; tk!($first:tt $($second:tt)? $($third:tt)?), $($remaining:tt)*) => {
            $res.push(tk!($first $($second)? $($third)?));
            munch_token!($lexer, $res; $($remaining)*)
        };
        ($lexer:expr, $res:expr; IntegerLiteral($val:literal), $($remaining:tt)*) => {
            $res.push(Token::IntegerLiteral($val));
            munch_token!($lexer, $res; $($remaining)*)
        };
        ($lexer:expr, $res:expr;) => ()
    }
    macro_rules! test_lex {
        ($text:expr, $($om:tt)*) => {{
            let arena = Bump::new();
            let mut lexer = PythonLexer::new(&arena, "");
            let mut res = Vec::new();
            munch_token!(lexer, res; $($om)*);
            assert_eq!(
                lexer.lex_all($text).unwrap(),
                res,
                "Unexpected parse for text:\n{}", $text
            );
        }}
    }
    macro_rules! lines {
        ($($line:literal),*) => {
            concat!($($line, "\n"),*)
        }
    }
    #[test]
    fn basic() {
        test_lex!(
            "def foo(a, b, cat)",
            Def, Ident("foo"), OpenParen,
            Ident("a"), Comma,
            Ident("b"), Comma, Ident("cat"),
            CloseParen,
        );
    }
    #[test]
    fn indentation() {
        test_lex!(
            lines!(
                r##"class Test:"##,
                r##"    first: int"##,
                r##""##,
                r##"    def chao(self):"##,
                r##"        pass"##,
                r##""##
            ),
            tk!(class), Ident("Test"), tk!(:), tk!('\n'),
            IncreaseIndent, Ident("first"), tk!(:),
            Ident("int"), tk!('\n'),
            tk!(def), Ident("chao"),
            tk!('('), Ident("self"), tk!(')'), tk!(:),
            tk!('\n'), IncreaseIndent, tk!(pass), Newline,
            DecreaseIndent, DecreaseIndent,
        );
        /*
         * This is the official example of 'confusing'
         * indentation from the docs:
         * https://docs.python.org/3.9/reference/lexical_analysis.html#indentation
         */
        test_lex!(
            r##"def perm(l):
        # Compute the list of all permutations of l
    if len(l) <= 1:
                  return [l]
    r = []
    for i in range(len(l)):
             s = l[:i] + l[i+1:]
             p = perm(s)
             for x in p:
              r.append(l[i:i+1] + x)
    return r"##,
            tk!(def), Ident("perm"), tk!('('), Ident("l"), tk!(')'),
            tk!(:), tk!('\n'), IncreaseIndent, tk!(if), Ident("len"), tk!('('),
            Ident("l"), tk!(')'), tk!(<=), IntegerLiteral(1), tk!(:), tk!('\n'),
            IncreaseIndent, tk!(return), tk!('['), Ident("l"), tk!(']'), tk!('\n'),
            DecreaseIndent, Ident("r"), tk!(=), tk!('['), tk!(']'), tk!('\n'),
            tk!(for), Ident("i"), tk!(in), Ident("range"), tk!('('),
                Ident("len"), tk!('('), Ident("l"), tk!(')'), tk!(')'),
                tk!(:), tk!('\n'), IncreaseIndent,
            Ident("s"), tk!(=), Ident("l"), tk!('['), tk!(:), Ident("i"),
                tk!(']'), tk!(+), Ident("l"), tk!('['), Ident("i"), tk!(+),
                IntegerLiteral(1), tk!(:), tk!(']'), Newline,
            Ident("p"), tk!(=), Ident("perm"), tk!('('), Ident("s"), tk!(')'), tk!('\n'),
            tk!(for), Ident("x"), tk!(in), Ident("p"), tk!(:), tk!('\n'),
            IncreaseIndent, Ident("r"), tk!(.), Ident("append"), tk!('('),
                Ident("l"), tk!('['), Ident("i"), tk!(:), Ident("i"), tk!(+),
                IntegerLiteral(1), tk!(']'), tk!(+), Ident("x"), tk!(')'), tk!('\n'),
            DecreaseIndent, DecreaseIndent,
            tk!(return), Ident("r"),
        )
    }
}
