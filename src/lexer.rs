//! A lexer for python-style source code
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::borrow::Borrow;
#[cfg(feature="num-bigint")]
use num_bigint::BigInt;
/// A rug BigInt
#[cfg(fefature="rug")]
pub type BigInt = rug::Integer;
#[cfg(not(any(feature="num-bigint"), feature="rug")))]
/// BigInts are unused
pub struct BigInt(!);

use logos::{Logos, Lexer};

/// A python identifier
///
/// These are interned, so there should
/// never be any duplicates within the same source file.
#[derive(Clone, Debug)]
pub struct Ident(Rc<str>);
impl Borrow<str> for Ident {
    #[inline]
    fn borrow(&self) -> &str {
        &*self.0
    }
}
impl Hash for Ident {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(Rc::as_ptr(&self.0), state);
    }
}
impl Eq for Ident {}
impl PartialEq for Ident {
    #[inline]
    fn eq(&self, other: &Ident) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
struct RawLexerState {
    starting_line: bool
}
impl Default for RawLexerState {
    fn default() -> RawLexerState {
        RawLexerState {
            // We start out beginning a line
            starting_line: true,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Keyword {
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
}

#[derive(Logos, Debug, PartialEq)]
#[logos(extras = RawLexerState)]
enum RawToken<'a> {
    // **************
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
    MinusEquals,
    #[token("*=")]
    StarEquals,
    #[token("/=")]
    SlashEquals,
    #[token("//=")]
    DoubleSlashEquals,
    #[token("%=")]
    PercentEquals,
    #[token("@=")]
    AtEquals,
    #[token("&=")]
    AndEquals,
    #[token("|=")]
    OrEquals,
    #[token("^=")]
    XorEquals,
    #[token(">>=")]
    RightShiftEquals,
    #[token("<<=")]
    LeftShiftEquals,
    #[token("**=")]
    DoubleStarEquals,
    // ********************
    //    Special Tokens
    // ********************
    /// A python identifier
    #[regex(r"[\p{XID_Start}_][\p{XID_Continue}_]*", ident)]
    Identifier(&'a str),
    #[regex(r"[1-9]([_]?[1-9])*|0([_]?0)*", parse_decimal_int)]
    Integer(Either<i64, BigInt>),
    /// A python string literal
    ///
    /// No backslashes or escapes have been interpreted
    /// in any way. It's up to the parser to do that.
    #[regex(r##"([rRuUfFbB]|[Ff][rR]|[rR][fFBb]|[Bb][rR])?(["']|"""|''')"##, lex_string)]
    String((StringInfo, &'a str)),
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
    #[regex(r"[\t ]+", raw_indent)]
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
struct StringInfo {
    /// An explicit string prefix,
    /// or none if it's just a plain string
    prefix: Option<StringPrefix>,
    /// True if the string is also a raw string
    raw: bool,
    quote_style: QuoteStyle
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
fn raw_indent<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> logos::Filter<usize> {
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
fn ident<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> &'a str {
    lex.slice()
}
fn parse_decimal_int<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> BigInt {
    // Eagerly attempt to parse as an `i64`
    match i64::try_from(lex.slice()) {
        Ok(val) => return Either::Left(val),
        Err(_) => {
            /*
             * There are two possible errors:
             * 1. Contains an underscore '_'
             * 2. Overflows
             *
             * Either way we want to fallback to
             * a more general implementation
             */
        }
    }
    #[cfg(feature="rug")]
    {
        let res = ::rug::Integer::parse(lex.slice()).unwrap();
        if let Some(i) = res.to_i64() {
            // Avoid wasting memory -> use i64 directly if possible
            return Either::Left(i);
        }
        Either::Right(res)
    }
    #[cfg(feature="num-bigint")]
    {
        // This is really our slow path
        let mut result = BigInt::new();
        for b in lex.bytes() {
            match b {
                b'0'..'9' => {
                    let digit_val = b - b'0';
                    result *= 10i64;
                    result += digit_val as i64;
                },
                b'_' => {}, // Ignore underscores
                _ => unreachable!("Invalid byte: {:?}", b)
            }
        }
        Either::Right(result)
    }
    #[cfg(not(any(feature="num-bigint", feature="rug")))]
    {
        unreachable!("BigInt failure: {}", lex.slice())
    }
}

fn lex_string<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> Result<(StringInfo, &'a str), StringError> {
    /*
     * Already parsed:
     * 1. Optionally: A prefix
     * 2. A string start, either triple string or single string
     */
    let info = {
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
        StringInfo { prefix, raw, quote_style }
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
        return Ok((info, lex.slice()));
    }
    Err(StringError::NoClosingQuote) // no closing paren
}
#[derive(Debug)]
enum StringError {
    NoClosingQuote,
    ForbiddenNewline
}
