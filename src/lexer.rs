//! A lexer for python-style source code
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::borrow::Borrow;

use logos::{Logos, Lexer};
use ahash::AHashSet;

/// A python identifier
///
/// These are interned, so there should
/// never be any duplicates within the same source file.
#[derive(Clone, Debug)]
pub struct Ident(Rc<str>);
impl Borrow<str> for Ident {
    #[inline]
    fn borrow(&self) -> &str {
        &**self.0
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
pub struct LexerState {
    idents: AHashSet<Ident>
}

#[derive(Logos, Debug, PartialEq)]
#[logos(extras = "LexerState")]
pub enum Token<'a> {
    // **************
    //    Keywords
    // **************
    #[token("False")]
    False,
    #[token("await")]
    Await,
    #[token("else")]
    Else,
    #[token("import")]
    Import,
    #[token("pass")]
    Pass,
    #[token("None")]
    None,
    #[token("True")]
    True,
    #[token("class")]
    Class,
    #[token("finally")]
    Finally,
    #[token("is")]
    Is,
    #[token("return")]
    Return,
    #[token("and")]
    And,
    #[token("continue")]
    Continue,
    #[token("for")]
    For,
    #[token("lambda")]
    Lambda,
    #[token("try")]
    Try,
    #[token("as")]
    As,
    #[token("def")]
    Def,
    #[token("from")]
    From,
    #[token("nonlocal")]
    Nonlocal,
    #[token("while")]
    While,
    #[token("assert")]
    Assert,
    #[token("del")]
    Del,
    #[token("global")]
    Global,
    #[token("not")]
    Not,
    #[token("with")]
    With,
    #[token("async")]
    Async,
    #[token("elif")]
    Elif,
    #[token("if")]
    If,
    #[token("or")]
    Or,
    #[token("yield")]
    Yield,
    /// A python identifier
    #[regex(r"[\p{XID_Start}_][\p{XID_Continue}_]*", ident)]
    Identifier(Ident),
    /// A python string literal
    ///
    /// No backslashes or escapes have been interpreted
    /// in any way. It's up to the parser to do that.
    #[regex(r##"([rRuUfFbB]|[Ff][rR]|[rR][fFBb]|[Bb][rR])?(["']|"""|''')"##, lex_string)]
    String((StringInfo, &'a str)),
    #[error]
    Error,
    // TODO: Indent/dedent
}

#[derive(Debug, Clone, Copy, Eq, ParitalEq)]
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
fn ident<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Ident {
    lex.extras.idents.get_or_insert_with(lex.slice(), Rc::from)
}
fn lex_string<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<(StringInfo, &'a str), ()> {
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
                (None, false);
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
        if remaining_bytes.get(index - 1) == Some(b'\\') {
            continue; // Skip escaped quote (or newline)
        }
        if remaining_bytes[index] == '\n' {
            if info.quote_style.is_triple_string() {
                continue; // just ignore the newline
            } else {
                // newline is an error...
                return Err(StringError::ForbiddenNewline);
            }
        }
        let end = if info.quote_style.is_triple_string() {
            match (info.quote_style, remaining_bytes.get(&[index..index + 3])) {
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
        } else { index + 1 };
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
