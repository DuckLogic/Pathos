use std::collections::VecDeque;

use crate::ast::Span;
use crate::lexer::{PythonLexer, Ident, LexError, Token};
use crate::alloc::Allocator;
use std::fmt::Display;
use std::ops::Deref;

#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    InvalidToken,
    AllocationFailed,
    UnexpectedEof,
    UnexpectedToken
}
#[derive(Debug, Clone)]
struct ParseErrorInner {
    span: Span,
    expected: Option<String>,
    actual: Option<String>,
    kind: ParseErrorKind,
}
#[derive(Debug, Clone)]
pub struct ParseError(Box<ParseErrorInner>);
impl ParseError {
    #[inline]
    pub fn builder(span: Span, kind: ParseErrorKind) -> ParseErrorBuilder {
        ParseErrorBuilder(ParseErrorInner {
            span, kind,
            expected: None, actual: None
        })
    }
}
pub struct ParseErrorBuilder(ParseErrorInner);
impl ParseErrorBuilder {
    #[inline]
    pub fn expected(self, f: impl ToString) -> Self {
        self.expected = Some(f.to_string());
        self
    }
    #[inline]
    pub fn actual(self, f: impl ToString) -> Self {
        self.actual = Some(f.to_string());
        self
    }
    #[inline]
    pub fn build(self) -> ParseError {
        ParseError(Box::new(ParseErrorInner {
            span: self.0.span,
            kind: self.0.kind,
            expected: self.0.expected.take(),
            actual: self.0.actual.take()
        }))
    }
}

pub struct SpannedToken<'a> {
    pub span: Span,
    pub kind: Token<'a>
}
impl<'a> Deref for SpannedToken<'a> {
    type Target = Token<'a>;
    #[inline]
    fn deref(&self) -> &Token<'a> {
        &self.kind
    }
}

pub struct Parser<'src, 'a> {
    arena: &'a Allocator,
    /// The buffer of look-ahead, containing
    /// tokens we have already lexed.
    ///
    /// This should always contain at least
    /// one token, unless we have reached the
    /// end of the file.
    ///
    /// These are ordered first token (peek) at the back,
    /// with the farthest ahead token at the front.
    ///
    /// To increase lookahead without consuming anything,
    /// add a new token with [VecDeque::push_front].
    ///
    /// To consume the first token, use [VecDeque::pop_back].
    buffer: VecDeque<SpannedToken<'a>>,
    lexer: PythonLexer<'src, 'a>,
}
impl<'a, 'src> Parser<'a, 'src> {
    #[inline]
    pub fn current_span(&self) -> Span {
        match self.buffer().back() {
            Some(tk) => tk.span,
            None => self.lexer.current_span()
        }
    }
    #[inline]
    pub fn peek(&self) -> Option<SpannedToken<'a>> {
        self.buffer.back().cloned()
    }
    fn assert_empty(&self) -> bool {
        debug_assert_eq!(
            self.lexer.next(),
            Ok(None)
        );
        true
    } 
    /// Look ahead the specified number of tokens
    ///
    /// If `amount == 0`, then this is equivalent
    /// to calling [Parser::peek]
    #[inline]
    pub fn look_ahead(&mut self, amount: usize) -> Result<Option<SpannedToken<'a>>, ParseError> {
        if amount >= self.buffer.len() {
            self.fill_buffer(amount + 1);
        }
        let index = self.buffer.len()
            .wrapping_sub(1)
            .wrapping_sub(amount);
        Ok(self.buffer.get(index).cloned())
    }
    /// Pop a token, lexing a new token and adding 
    /// it to the internal buffer
    ///
    /// Returns an error if the lexer
    /// has an issue with the next token.
    /// The current token *cannot* trigger an error,
    /// because it's already in the buffer.
    #[inline]
    pub fn pop(&mut self) -> Result<Option<SpannedToken<'a>>, ParseError> {
        match self.buffer.len() {
            0 => {
                /*
                 * The buffer should only be
                 * empty at EOF. In debug mode,
                 * we should double-check this
                 */
                debug_assert!(self.assert_empty());
                return Ok(None)
            },
            1 => {
                /*
                 * We only have one left,
                 * so consuming this token would
                 * otherwise empty the buffer.
                 * Maintain our invariant of having,
                 * by requesting to fill the buffer to
                 * length two.
                 */
                self.fill_buffer(2);
                #[cfg(debug_assertions)] {
                    match self.buffer.len() {
                        0 => unreachable!(),
                        1 => debug_assert!(self.assert_empty()),
                        _ => {}
                    }
                }
            },
            _ => {},
        }
        Ok(self.buffer.pop_front())
    }
    #[cold]
    fn fill_buffer(&mut self, amount: usize) -> Result<bool, ParseError> {
        // TODO: Should we buffer more aggressively?
        self.buffer.reserve(amount);
        while self.buffer.len() < amount {
            let lexer_span = self.lexer.current_span();
            self.push_front(match self.lexer.next() {
                Ok(Some(val)) => SpannedToken {
                    span: lexer_span, kind: val
                },
                Ok(None) => return Ok(false),
                Err(cause) => {
                    let kind = match cause {
                        LexError::AllocFailed => {
                            // TODO: Handle OOM without boxing errors.....
                            ParseErrorKind::AllocationFailed
                        },
                        LexError::InvalidToken => ParseErrorKind::InvalidToken
                    };
                    return Err(ParseError::builder(
                        lexer_span, kind
                    ).build());
                },
            })
        }
        Ok(true)
    }
    /// Check if the parser is finished
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.buffer.is_empty()
    }
    //
    // Utilities
    //
    pub fn expect(&mut self, expected: Token<'a>)  -> Result<(), ParseError> {
        self.expect_if(&expected, |actual| *actual == expected)
    }
    pub fn expect_if(
        &mut self, expected: &dyn Display,
        func: impl FnOnce(&Token<'a>) -> bool
    ) -> Result<(), ParseError> {
        self.expect_map(expected, |token| if func(token) { Some(()) } else { None })
    }
    pub fn expect_map<T>(
        &mut self, expected: &dyn Display,
        func: impl FnOnce(&Token<'a>) -> Option<T>
    ) -> Result<T, ParseError> {
        match self.peek() {
            Some(actual) => {
                if let Some(res) = func(&actual) {
                    match self.pop()? {
                        Ok(tk) => {
                            debug_assert_eq!(tk, actual);
                            return Ok(res);
                        },
                        None => unreachable!()
                    }
                }
            },
            None => {}
        }
        // fallthrough to error
        Err(self.unexpected(expected))
    }
    #[cold]
    pub fn unexpected(&self, expected: &dyn Display) -> ParseError {
        let kind = if self.is_finished() {
            ParseError::UnexpectedEof
        } else {
            ParseError::UnexpectedToken
        };
        ParseError::builder(self.current_span(), kind)
            .expected(expected)
            .actual(match self.peek() {
                Some(tk) => format!("{}", tk),
                None => "EOF".to_string()
            })
            .build()
    }
    pub fn pop_ident(&mut self) -> Result<&'a Ident<'a>, ParseError> {
        self.expect_with("an identifier", |token| match token {
            &Token::Ident(ident) => Some(ident),
            _ => None
        })
    }
}