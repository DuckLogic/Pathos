//! A lexer for python-style source code
use std::fmt::{self, Formatter, Display, Debug};
use std::hash::{Hash, Hasher};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::num::ParseIntError;
use std::backtrace::Backtrace;
use std::error::Error;

use logos::{Logos, Lexer};
use either::Either;
use thiserror::Error;

use pathos::errors::{SpannedError, ErrorSpan};
use pathos::ast::Span;
use pathos::ast::ident::{SymbolTable, Symbol};
use pathos::lexer::{Token as IToken, TokenKind as ITokenKind, Lexer as ILexer, LexerError as ILexerError};
use pathos::alloc::{Allocator, AllocError};

use pathos_python_ast::constants::StringLiteral;
use pathos_python_ast::constants::{BigInt, QuoteStyle, StringPrefix, StringStyle};
use pathos_python_ast::tree::{UnaryOp, Operator, BoolOp, ComparisonOp};
use pathos_unicode::NameResolveError;

pub type SpannedToken<'a> = pathos::lexer::SpannedToken<'a, Token<'a>>;

pub trait OperatorExt: Sized {
    fn from_token(tk: &Token) -> Result<Self, FromTokenError>;
    #[inline]
    fn token<'a>(&self) -> Option<Token<'a>> {
        None
    }
}
impl OperatorExt for BoolOp {
    #[inline]
    fn from_token(tk: &Token) -> Result<BoolOp, FromTokenError> {
        Ok(match *tk {
            Token::And => BoolOp::And,
            Token::Or => BoolOp::Or,
            _ => return Err(FromTokenError::CompletelyInvalid)
        })
    }
    #[inline]
    fn token<'a>(&self) -> Option<Token<'a>> {
        Some(match self {
            BoolOp::And => Token::And,
            BoolOp::Or => Token::Or
        })
    }
}
impl OperatorExt for Operator {
    #[inline]
    fn from_token(tk: &Token) -> Result<Self, FromTokenError> {
        Ok(match *tk {
            Token::Plus => Operator::Add,
            Token::Minus => Operator::Sub,
            Token::Star => Operator::Mult,
            Token::At => Operator::MatMult,
            Token::Slash => Operator::Div,
            Token::DoubleSlash => Operator::FloorDiv,
            Token::DoubleStar => Operator::Pow,
            Token::LeftShift => Operator::LShift,
            Token::RightShift => Operator::RShift,
            Token::BitwiseOr => Operator::BitOr,
            Token::BitwiseXor => Operator::BitXor,
            Token::Ampersand => Operator::BitAnd,
            _ => return Err(FromTokenError::CompletelyInvalid)
        })
    }
}
impl OperatorExt for UnaryOp {
    #[inline]
    fn from_token(tk: &Token) -> Result<Self, FromTokenError> {
        Ok(match *tk {
            Token::BitwiseInvert => UnaryOp::Invert,
            Token::Not => UnaryOp::Not,
            Token::Plus => UnaryOp::UAdd,
            Token::Minus => UnaryOp::USub,
            _ => return Err(FromTokenError::CompletelyInvalid)
        })
    }
}
impl OperatorExt for ComparisonOp {
    #[inline]
    fn from_token(tk: &Token) -> Result<Self, FromTokenError> {
        Ok(match *tk {
            Token::DoubleEquals => ComparisonOp::Eq,
            Token::NotEquals => ComparisonOp::NotEq,
            Token::LessThan => ComparisonOp::Lt,
            Token::LessThanOrEqual => ComparisonOp::LtE,
            Token::GreaterThan => ComparisonOp::Gt,
            Token::GreaterThanOrEqual => ComparisonOp::GtE,
            Token::In => ComparisonOp::In,
            /*
             * Could be 'is', 'is not', or 'not in'
             * but we don't know anything for sure at this point
             */
            Token::Is | Token::Not => return Err(FromTokenError::Ambiguous),
            _ => return Err(FromTokenError::CompletelyInvalid)
        })
    }
}
#[derive(Copy, Clone, Debug)]
pub enum FromTokenError {
    /// Indicates that the list of provided tokens
    /// is completely invalid (at least the start)
    CompletelyInvalid,
    /// Indicates that the list of provided tokens
    /// is of insufficient length to make a conclusive determination
    Ambiguous
}

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


/// A python identifier
///
/// TODO: Any further unification with [Symbol](crate::ast::Symbol)
#[derive(Copy, Clone)]
pub struct Ident<'a> {
    /*
     * TODO: Store precomputed hashes or something?
     * I feel like it might be helpful to have some sort of `IdentMap`.
     * However right now, that's probably just premature optimization.
     */
    text: &'a str,
}
impl Ident<'static> {
    /// Create an identifier from the specified static text
    #[inline]
    pub fn from_static_text(text: &'static str) -> Ident<'static> {
        Ident { text }
    }
}
impl<'a> Ident<'a> {
    #[inline]
    pub fn text(&self) -> &'a str {
        self.text
    }
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
impl Debug for Ident<'_> {
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
#[derive(Debug, PartialEq, Error)]
pub enum LexError {
    #[error("Invalid token")]
    InvalidToken {
        span: Span,
    },
    #[error("Allocation failed")]
    AllocFailed,
    #[error("Invalid string: {0}")]
    InvalidString(#[source] StringError)
}
impl SpannedError for LexError {
    fn span(&self) -> ErrorSpan {
        match *self {
            LexError::InvalidToken { span, .. } => ErrorSpan::Span(span),
            LexError::AllocFailed => ErrorSpan::AllocationFailed,
            LexError::InvalidString(ref cause) => cause.span(),
        }
    }
}
impl ILexerError for LexError {
    fn upcast(&self) -> &(dyn Error + 'static) {
        self
    }

    fn cast_alloc_failed(&self) -> Option<&AllocError> {
        match *self {
            LexError::InvalidToken { .. } |
            LexError::InvalidString(_) => None,
            LexError::AllocFailed => Some(&AllocError)
        }
    }
}
impl From<AllocError> for LexError {
    #[inline]
    #[cold]
    fn from(_e: AllocError) -> LexError {
        LexError::AllocFailed
    }
}
impl From<StringError> for LexError {
    #[cold]
    fn from(cause: StringError) -> LexError {
        match cause.kind {
            StringErrorKind::AllocFailed => LexError::AllocFailed,
            _ => LexError::InvalidString(cause)
        }
    }
}

pub struct PythonLexer<'src, 'arena> {
    arena: &'arena Allocator,
    raw_lexer: Lexer<'src, RawToken<'src>>,
    symbol_table: SymbolTable<'arena>,
    pending_indentation_change: isize,
    indent_stack: Vec<usize>,
    starting_line: bool,
    relative_buffer_position: u64,
}
impl<'src, 'arena> Debug for PythonLexer<'src, 'arena> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let span = self.current_span();
        let slice = self.raw_lexer.slice();
        // TODO: More fields?
        f.debug_struct("PythonLexer")
            .field("span", &span)
            .field("slice", &slice)
            .finish()
    }
}

macro_rules! translate_tokens {
    ($target:expr; $($name:ident $( ( $mtch:ident ) )? $(=> $handler:expr)?),*) => {{
        match $target {
            $(RawToken::$name $( ( $mtch ) )? => {
                translate_tokens!(handler for $name $( ($mtch) )? $(=> $handler)?)
            }),*
        }
    }};
    (handler for $name:ident $(( $mtch:ident ) )? => $handler:expr) => ($handler);
    (handler for $name:ident) => (Token::$name);
}
impl<'src, 'arena, 'sym> PythonLexer<'src, 'arena> {
    /*
     * TODO: Encapsulate and support buffered input
     *
     * Unfortunately, input to logos currently can't be buffered with Read.
     * See issue #159 for details
     */
    pub fn original_text(&self) -> &'src str {
        self.raw_lexer.source()
    }
    pub fn new(arena: &'arena Allocator, symbol_table: SymbolTable<'arena>, text: &'src str) -> Self {
        PythonLexer {
            arena,
            raw_lexer: RawToken::lexer(text),
            symbol_table,
            pending_indentation_change: 0,
            indent_stack: vec![0],
            starting_line: true,
            relative_buffer_position: 0,
        }
    }
    pub fn reset(&mut self, text: &'src str) {
        self.raw_lexer = RawToken::lexer(text);
        self.pending_indentation_change = 0;
        self.indent_stack.clear();
        self.indent_stack.push(0);
        self.relative_buffer_position = 0;
        self.starting_line = true;
    }
    pub fn lex_all(&mut self, text: &'src str) -> Result<Vec<Token<'arena>>, LexError> {
        self.reset(text);
        let mut res = Vec::new();
        while let Some(token) = self.try_next()? {
            res.push(token);
        }
        Ok(res)
    }
    #[inline]
    pub fn current_span(&self) -> Span {
        let raw = self.raw_lexer.span();
        let offset = self.relative_buffer_position;
        Span {
            start: offset.saturating_add(raw.start as u64).into(),
            end: offset.saturating_add(raw.end as u64).into()
        }
    }
    pub fn create_ident(&mut self, text: &'src str) -> Result<Symbol<'arena>, AllocError> {
        self.symbol_table.alloc(text)
    }
    #[allow(unused)]
    pub fn try_next(&mut self) -> Result<Option<Token<'arena>>, LexError> {
        match self.pending_indentation_change.cmp(&0) {
            Ordering::Equal => {},
            Ordering::Less => {
                self.pending_indentation_change += 1;
                return Ok(Some(Token::DecreaseIndent));
            },
            Ordering::Greater => {
                self.pending_indentation_change -= 1;
                return Ok(Some(Token::IncreaseIndent));
            }
        }
        'yum: loop {
            if self.starting_line {
                self.starting_line = false;
                let amount = self.raw_lexer.remainder().bytes()
                    .position(|b| b != b' ' && b != b'\t')
                    .unwrap_or(0);
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
                match amount.cmp(&current_top) {
                    Ordering::Greater => {
                        self.indent_stack.push(amount);
                        return Ok(Some(Token::IncreaseIndent));
                    },
                    Ordering::Less => {
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
                    },
                    Ordering::Equal => {}
                }
            }
            // Lets do some post processing ;)
            let token = match self.raw_lexer.next() {
                Some(val) => val,
                None => return Ok(None) // EOF
            };
            return Ok(Some(translate_tokens!(token;
                EscapedNewline => {
                    continue 'yum;
                },
                RawNewline => {
                    self.starting_line = true;
                    // TODO: Do we need to do anything else?
                    Token::Newline
                },
                Integer (inner) => {
                    match inner {
                        Either::Left(int) => Token::IntegerLiteral(int),
                        Either::Right(int) => Token::BigIntegerLiteral(self.arena.alloc(int)?)
                    }
                },
                Identifier (text) => {
                    Token::Ident(self.create_ident(text)?)
                },
                FloatLiteral (f) => Token::FloatLiteral(f),
                Error => {
                    return Err(LexError::InvalidToken { span: self.current_span() })
                },
                StartString(style) => {
                    Token::StringLiteral(self.lex_string(style)?)
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
    fn lex_string(&mut self, style: StringStyle) -> Result<&'arena StringLiteral<'arena>, StringError> {
        // Estimate the size of the string
        let originally_remaining = self.raw_lexer.remainder();
        let original_bytes = originally_remaining.as_bytes();
        let string_start = self.current_span().start;
        let estimated_size = if style.quote_style.is_triple_string() {
            ::memchr::memmem::find_iter(
                original_bytes,
                style.quote_style.text().as_bytes(),
            ).find(|&index| index == 0 || original_bytes.get(index - 1) != Some(&b'\\'))
        } else {
            ::memchr::memchr_iter(
                style.quote_style.start_byte(),
                original_bytes
            ).find(|&index| index == 0 || original_bytes.get(index - 1) != Some(&b'\\'))
        }.ok_or_else(|| {
            let span = string_start.with_len(style.quote_style.len());
            StringError {
                kind: StringErrorKind::MissingEnd,
                span: span.into(), backtrace: Backtrace::capture(),
                entire_string_span: None
            }
        })?;
        // TODO: Is the 'estimated_size' ever incorrect?
        let entire_string_span = string_start.with_len(estimated_size);
        let mut buffer = pathos::alloc::String::with_capacity(
            self.arena,
            estimated_size
        )?;
        let mut sub_lexer = StringPart::lexer(
            self.raw_lexer.remainder()
        );
        while let Some(tk) = sub_lexer.next() {
            let relative_span = Span::from_logos_span(sub_lexer.span());
            let span = relative_span.offset(entire_string_span.start.0);
            let give_error = |span: Span, kind: StringErrorKind| {
                StringError {
                    kind, backtrace: Backtrace::capture(),
                    span: span.into(), entire_string_span: Some(entire_string_span)
                }
            };
            match tk.interpret() {
                InterpretedStringPart::RegularText(text) => {
                    buffer.push_str(text)?;
                }
                InterpretedStringPart::EscapedLiteral(e) => {
                    if style.raw {
                        // Raw strings don't interpret escapes
                        buffer.push_str(sub_lexer.slice())?;
                    } else {
                        buffer.push(e)?;
                    }
                },
                InterpretedStringPart::EscapedLineEnd => {
                    /*
                     * "example \
                     * continued" ->
                     * Intentionally ignore the line
                     * ending
                     */
                },
                InterpretedStringPart::UnescapedLineEnd => {
                    if style.quote_style.is_triple_string() {
                        buffer.push('\n')?;
                    } else {
                        self.raw_lexer.bump(relative_span.start.0 as usize);
                        return Err(give_error(span.start.with_len(1), StringErrorKind::ForbiddenNewline));
                    }
                },
                InterpretedStringPart::NamedEscape(name) => {
                    if style.raw {
                        buffer.push_str(sub_lexer.slice())?;
                        continue;
                    }
                    let upper = name.to_uppercase();
                    match pathos_unicode::resolve_name(&upper) {
                        Ok(c) => buffer.push(c)?,
                        Err(reason) => {
                            self.raw_lexer.bump(relative_span.start.0 as usize);
                            return Err(give_error(span, match reason {
                                NameResolveError::Unsupported => StringErrorKind::UnsupportedNamedEscape,
                                NameResolveError::UnknownName => StringErrorKind::InvalidNamedEscape
                            }))
                        }
                    }
                },
                InterpretedStringPart::UnescapedQuote(quote) => {
                    match (style.quote_style, quote) {
                        /*
                         * Encountering a triple quote while parsing a single
                         * a single quote is actually *not* an error:
                         * "a""" -> "a" + "" -> "a"
                         * In other words, we can treat "a"""
                         * closing a single quote just like "a"
                         */
                        (QuoteStyle::Single, QuoteStyle::SingleLong) |
                        (QuoteStyle::Double, QuoteStyle::DoubleLong) |
                        (QuoteStyle::Single, QuoteStyle::Single) |
                        (QuoteStyle::Double, QuoteStyle::Double) |
                        (QuoteStyle::SingleLong, QuoteStyle::SingleLong) |
                        (QuoteStyle::DoubleLong, QuoteStyle::DoubleLong) => {
                            // We've encountered our closing
                            self.raw_lexer.bump(relative_span.start.index() + quote.len());
                            return Ok(&*self.arena.alloc(StringLiteral {
                                style, value: buffer.into_str()
                            })?)
                        },
                        (QuoteStyle::Double, QuoteStyle::Single) |
                        (QuoteStyle::Double, QuoteStyle::SingleLong) |
                        (QuoteStyle::Single, QuoteStyle::Double) |
                        (QuoteStyle::Single, QuoteStyle::DoubleLong) |
                        /*
                         * A double or triple
                         * quote encountering anything
                         * other than its corresponding 
                         * closing effectively ignores the quote
                         */
                        (QuoteStyle::DoubleLong, _) |
                        (QuoteStyle::SingleLong, _) => {
                            buffer.push_str(sub_lexer.slice())?;
                        },
                    }
                },
                InterpretedStringPart::Error => {
                    self.raw_lexer.bump(relative_span.start.index());
                    let mut chrs = sub_lexer.slice().chars();
                    if let Some('\\') = chrs.next() {
                        if let Some(next) = chrs.next() {
                            return Err(give_error(span, StringErrorKind::InvalidEscape { c: next }))
                        }
                    }
                    return Err(give_error(span, StringErrorKind::MissingEnd))
                }
            }
        }
        Err(StringError {
            entire_string_span: None,
            span: entire_string_span.into(),
            kind: StringErrorKind::MissingEnd,
            backtrace: Backtrace::capture()
        })
    }
    #[inline]
    pub fn into_symbol_table(self) -> SymbolTable<'arena> {
        self.symbol_table
    }
}
impl<'src, 'a> ILexer<'a> for PythonLexer<'src, 'a> {
    type Error = LexError;
    type Token = Token<'a>;

    fn original_text(&self) -> &str {
        self.raw_lexer.source()
    }

    fn current_span(&self) -> Span {
        self.current_span()
    }

    fn try_next(&mut self) -> Result<Option<Self::Token>, Self::Error> {
        self.try_next()
    }
}


#[derive(Copy, Clone, PartialEq)]
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
    // ***************
    //    Special Tokens
    // ***************
    IntegerLiteral(i64),
    FloatLiteral(f64),
    /// An arbitrary-precision integer,
    /// that is too big to fit in a regular int.
    BigIntegerLiteral(&'arena BigInt),
    /// A string literal
    // TODO: Actually parse the underlying text
    StringLiteral(&'arena StringLiteral<'arena>),
    /// An identifier.
    ///
    /// Do not confuse this with [Token::IncreaseIndent].
    /// This has the shorter name because it is more common.
    Ident(Symbol<'arena>),
    /// Increase the indentation
    IncreaseIndent,
    /// Decrease the indentation
    DecreaseIndent,
    /// A logical newline
    Newline,
}
impl<'a> IToken<'a> for Token<'a> {
    type Kind = TokenKind;

    fn kind(&self) -> Self::Kind {
        todo!()
    }

    #[inline]
    fn is_newline(&self) -> bool {
        matches!(*self, Token::Newline)
    }
}
#[derive(Copy, Clone, Debug)]
pub enum TokenKind {}
impl ITokenKind for TokenKind {

}
impl Display for TokenKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {}
    }
}
impl<'a> Token<'a> {
    /// If this token is an augmented assignment operator (`+=`, `*=`, etc),
    /// then get the corresponding [Operator]
    pub fn aug_assign_op(&self) -> Option<Operator> {
        match *self {
            Token::PlusEquals => Some(Operator::Add),
            Token::MinusEquals => Some(Operator::Sub),
            Token::StarEquals => Some(Operator::Mult),
            Token::AtEquals => Some(Operator::MatMult),
            Token::SlashEquals => Some(Operator::Div),
            Token::DoubleSlashEquals => Some(Operator::FloorDiv),
            Token::PercentEquals => Some(Operator::Mod),
            Token::DoubleStarEquals => Some(Operator::Pow),
            Token::RightShiftEquals => Some(Operator::RShift),
            Token::LeftShiftEquals => Some(Operator::LShift),
            Token::AndEquals => Some(Operator::BitAnd),
            Token::XorEquals => Some(Operator::BitXor),
            Token::OrEquals => Some(Operator::BitOr),
            _ => None
        }
    }
    #[inline]
    fn static_text(&self) -> Option<&'static str> {
        Some(match *self {
            Token::False => "False",
            Token::Await => "await",
            Token::Else => "else",
            Token::Import => "import",
            Token::Pass => "pass",
            Token::None => "None",
            Token::True => "True",
            Token::Class => "class",
            Token::Finally => "finally",
            Token::Is => "is",
            Token::Return => "return",
            Token::And => "and",
            Token::Continue => "continue",
            Token::For => "for",
            Token::Lambda => "lambda",
            Token::Try => "try",
            Token::As => "as",
            Token::Def => "def",
            Token::From => "from",
            Token::Nonlocal => "nonlocal",
            Token::While => "while",
            Token::Assert => "assert",
            Token::Del => "del",
            Token::Global => "global",
            Token::Not => "not",
            Token::With => "with",
            Token::Async => "async",
            Token::Elif => "elif",
            Token::If => "if",
            Token::Or => "or",
            Token::Yield => "yield",
            Token::Break => "break",
            Token::Except => "except",
            Token::In => "in",
            Token::Raise => "raise",
            Token::Plus => "+",
            Token::Minus => "-",
            Token::Star => "*",
            Token::DoubleStar => "**",
            Token::Slash => "/",
            Token::DoubleSlash => "//",
            Token::Percent => "%",
            Token::At => "@",
            Token::LeftShift => "<<",
            Token::RightShift => ">>",
            Token::Ampersand => "&",
            Token::BitwiseOr => "|",
            Token::BitwiseXor => "^",
            Token::BitwiseInvert => "~",
            Token::AssignOperator => ":=",
            Token::LessThan => "<",
            Token::GreaterThan => ">",
            Token::LessThanOrEqual => "<=",
            Token::GreaterThanOrEqual => ">=",
            Token::DoubleEquals => "==",
            Token::NotEquals => "!=",
            Token::OpenParen => "(",
            Token::CloseParen => ")",
            Token::OpenBracket => "[",
            Token::CloseBracket => "]",
            Token::OpenBrace => "{",
            Token::CloseBrace => "}",
            Token::Comma => ",",
            Token::Colon => ":",
            Token::Period => ".",
            Token::Semicolon => ";",
            Token::Equals => "=",
            Token::Arrow => "->",
            Token::PlusEquals => "+=",
            Token::MinusEquals => "-=",
            Token::StarEquals => "*=",
            Token::SlashEquals => "/=",
            Token::DoubleSlashEquals => "//=",
            Token::PercentEquals => "%=",
            Token::AtEquals => "@=",
            Token::AndEquals => "&=",
            Token::OrEquals => "|=",
            Token::XorEquals => "^=",
            Token::RightShiftEquals => ">>=",
            Token::LeftShiftEquals => "<<=",
            Token::DoubleStarEquals => "**=",
            _ => return None
        })
    }
    fn fmt_impl(&self, f: &mut Formatter<'_>, escaped: bool) -> fmt::Result {
        use std::fmt::Write;
        if let Some(text) = self.static_text() {
            if escaped {
                write!(f, "{:?}", text)
            } else {
                f.write_str(text)
            }
        } else {
            let should_write_quotes = escaped && !matches!(*self, Token::StringLiteral(_));
            if should_write_quotes {
                f.write_char('"')?;
            }
            match *self {
                Token::IntegerLiteral(val) => {
                    write!(f, "{}", val)
                }
                Token::FloatLiteral(flt) => {
                    write!(f, "{}", flt)
                },
                Token::BigIntegerLiteral(big_int) => {
                    write!(f, "{}", big_int)
                },
                Token::StringLiteral(lit) => {
                    if escaped {
                        write!(f, "String({})", lit)
                    } else {
                        write!(f, "{}", lit)
                    }
                }
                Token::Ident(id) => {
                    write!(f, "{}", id.text())
                },
                Token::IncreaseIndent => {
                    write!(f, "INDENT")
                },
                Token::DecreaseIndent => {
                    write!(f, "DEDENT")
                },
                Token::Newline => {
                    write!(f, "NEWLINE")
                },
                _ => {
                    /*
                     * This is logically unreachable,
                     * because we should've handled
                     * all operators and keywords in 'static_text'.
                     * We handle everything else in the match statement.
                     * However, the compiler can't prove this.
                     * Just write this as a fallback,
                     * since a panic would be unhelpful.
                     */
                    write!(f, "Token::Unreachable({:?})", self)
                }
            }?;
            if should_write_quotes {
                f.write_char('"')?;
            }
            Ok(())
        }
    }
}
impl<'a> Debug for Token<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_impl(f, true)
    }
}
impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.fmt_impl(f, false)
    }
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
    #[regex(r##"([rRuUfFbB]|[Ff][rR]|[rR][fFBb]|[Bb][rR])?(["']|"""|''')"##, lex_string_start)]
    StartString(StringStyle),
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
    RawNewline,
    #[regex(r"\\(\n|\r\n)", logos::skip)]
    EscapedNewline
}

fn skip_comment<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> logos::Skip {
    debug_assert_eq!(lex.slice(), "#");
    if let Some(line_end) = ::memchr::memchr(b'\n', lex.remainder().as_bytes()) {
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
fn fallback_parse_int(
    radix: i64, text: &str
) -> BigInt {
    use pathos_python_ast::constants::BigIntInternal;
    /*
     * We get here if the regular integer parse code overflows.
     *
     * We fallback to parsing as a arbitrary precision integer.
     * NOTE: This conflicts with an inherent method when using 'rug',
     * so we need to disambiguate.
     */
    BigIntInternal::parse_radix(radix as u32, text).unwrap()
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
fn lex_string_start<'a>(lex: &mut Lexer<'a, RawToken<'a>>) -> StringStyle {
    /*
     * Already parsed:
     * 1. Optionally: A prefix
     * 2. A string start, either triple string or single string
     */
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
}

#[derive(Logos, Debug)]
enum StringPart<'src> {
    #[token(r#"\\"#)]
    EscapedBackslash,
    #[token(r#"\'"#)]
    EscapedSingleQuote,
    #[token(r#"\""#)]
    EscapedDoubleQuote,
    #[token(r#"\a"#)]
    EscapedAsciiBell,
    #[token(r#"\b"#)]
    EscapedBackspace,
    #[token(r#"\f"#)]
    EscapedFormFeed,
    #[token(r#"\n"#)]
    EscapedNewline,
    #[token(r#"\r"#)]
    EscapedCarriageReturn,
    #[token(r#"\t"#)]
    EscapedTab,
    #[token(r#"\v"#)]
    EscapedVerticalTab,
    #[regex(r#"\\[0-7][0-7]?[0-7]?"#, parse_octal_escape)]
    OctalEscape(u8),
    #[regex(r#"\\x[0-9A-Fa-f][0-9A-Fa-f]"#, parse_hex_escape)]
    HexEscape(u8),
    #[token("\\\n")]
    EscapedLineEnd,
    #[token("\n")]
    UnescapedLineEnd,
    #[regex(r#"\\N\{\w+\}"#)]
    NamedEscape(&'src str),
    #[token(r#"""#)]
    DoubleQuote,
    #[token(r#"'"#)]
    SingleQuote,
    #[token(r###"""""###)]
    TripleDoubleQuote,
    #[token(r###"'''"###)]
    TripleSingleQuote,
    #[regex(r#"[^\\\n"']+"#)]
    RegularText(&'src str),
    #[error]
    Error
}
impl<'src> StringPart<'src> {
    /// Collapse this token into a more manageable form
    #[inline]
    fn interpret(self) -> InterpretedStringPart<'src> {
        InterpretedStringPart::EscapedLiteral(match self {
            StringPart::EscapedBackslash => '\\',
            StringPart::EscapedSingleQuote => '\'',
            StringPart::EscapedDoubleQuote => '\"',
            StringPart::EscapedBackspace => '\\',
            StringPart::EscapedAsciiBell => '\u{07}',
            StringPart::EscapedFormFeed => '\u{0C}',
            StringPart::EscapedTab => '\t',
            StringPart::EscapedCarriageReturn => '\r',
            StringPart::EscapedVerticalTab => '\u{0A}',
            StringPart::EscapedNewline => '\n',
            StringPart::OctalEscape(val) => val as char,
            StringPart::HexEscape(val) => val as char,
            StringPart::EscapedLineEnd => {
                return InterpretedStringPart::EscapedLineEnd
            },
            StringPart::UnescapedLineEnd => {
                return InterpretedStringPart::UnescapedLineEnd
            },
            StringPart::NamedEscape(name) => {
                return InterpretedStringPart::NamedEscape(name)
            },
            StringPart::DoubleQuote => {
                return InterpretedStringPart::UnescapedQuote(QuoteStyle::Double)
            },
            StringPart::SingleQuote => {
                return InterpretedStringPart::UnescapedQuote(QuoteStyle::Single)
            },
            StringPart::TripleDoubleQuote => {
                return InterpretedStringPart::UnescapedQuote(QuoteStyle::DoubleLong)
            },

            StringPart::TripleSingleQuote => {
                return InterpretedStringPart::UnescapedQuote(QuoteStyle::SingleLong)
            },
            StringPart::RegularText(text) => {
                return InterpretedStringPart::RegularText(text)
            }
            StringPart::Error => {
                return InterpretedStringPart::Error
            }
        })
    }
}
enum InterpretedStringPart<'src> {
    EscapedLiteral(char),
    NamedEscape(&'src str),
    EscapedLineEnd,
    UnescapedLineEnd,
    UnescapedQuote(QuoteStyle),
    RegularText(&'src str),
    Error
}
fn parse_octal_escape<'a>(lex: &mut Lexer<'a, StringPart<'a>>) -> Result<u8, ParseIntError> {
    debug_assert!(lex.slice().starts_with('\\'));
    u8::from_str_radix(&lex.slice()[1..], 8)
}
fn parse_hex_escape<'a>(lex: &mut Lexer<'a, StringPart<'a>>) -> u8 {
    debug_assert!(lex.slice().starts_with('\\'));
    u8::from_str_radix(&lex.slice()[1..], 16).unwrap()
}
#[derive(Debug)]
pub struct StringError {
    /// The span of the entire string, if that differs from the primary span of this error
    ///
    /// This may be `None` even if the primary of the span is `Some`
    pub entire_string_span: Option<Span>,
    /// The span of the error, or `None` if this is an allocation-failed error
    pub span: ErrorSpan,
    pub kind: StringErrorKind,
    backtrace: Backtrace,
}
impl SpannedError for StringError {
    #[inline]
    fn span(&self) -> ErrorSpan {
        self.span
    }
}
impl Error for StringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.kind.source()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        Some(&self.backtrace)
    }

}
impl PartialEq for StringError {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
impl Display for StringError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.kind, f)
    }
}
#[derive(Debug, Clone, PartialEq, Error)]
pub enum StringErrorKind {
    #[error("Missing end of quote")]
    MissingEnd,
    #[error("Newlines are forbidden inside strings")]
    ForbiddenNewline,
    #[error("Invalid escape char: {c:?}")]
    InvalidEscape {
        c: char,
    },
    #[error("Invalid named escape")]
    InvalidNamedEscape,
    /// Indicates that named escapes are unsupported,
    /// because the crate was compiled without full unicode support.
    #[error("Named escapes are unsupported")]
    UnsupportedNamedEscape,
    #[error("Allocation failed")]
    AllocFailed
}
impl From<AllocError> for StringError {
    #[cold]
    fn from(_cause: AllocError) -> StringError {
        StringError {
            kind: StringErrorKind::AllocFailed,
            backtrace: Backtrace::capture(),
            span: ErrorSpan::AllocationFailed,
            entire_string_span: None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use bumpalo::Bump;
    #[cfg(test)]
    use pretty_assertions::{assert_eq,};
    macro_rules! munch_token {
        ($lexer:expr, $res:expr; Ident($text:literal), $($remaining:tt)*) => {
            $res.push(Token::Ident($lexer.create_ident($text).unwrap()));
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
        ($text:expr, $($om:tt)*) => { #[allow(clippy::vec_init_then_push)] {
            let arena = Allocator::new(Bump::new());
            let symbol_table = SymbolTable::new(&arena);
            let mut lexer = PythonLexer::new(&arena, symbol_table, "");
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
        test_lex!(
            "[1, 2, 3]",
            OpenBracket, IntegerLiteral(1),
            Comma, IntegerLiteral(2), Comma,
            IntegerLiteral(3), CloseBracket,
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
