use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::ast::Span;
use crate::lexer::{PythonLexer, Ident, LexError, Token};
use crate::alloc::{Allocator, AllocError};
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
    /// The span of the source location
    ///
    /// This is `None` if an out of
    /// memory error occurs, otherwise
    /// it must be present
    span: Option<Span>,
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
            span: Some(span), kind,
            expected: None, actual: None
        })
    }
}
impl From<AllocError> for ParseError {
    #[cold]
    fn from(cause: AllocError) -> ParseError {
        /*
         * This is a bit of a conondrum.
         * We're allocating memory for the error value
         * even though we have an 'out of memory'
         * condition.
         * Realistically though, we'll probably only encounter
         * OOM if the counter hits the internal limit.
         */
        ParseError(Box::new(ParseErrorInner {
            span: None,
            expected: None,
            actual: None,
            kind: ParseErrorKind::AllocationFailed
        }))
    }
}
pub struct ParseErrorBuilder(ParseErrorInner);
impl ParseErrorBuilder {
    #[inline]
    pub fn expected(self, f: impl ToString) -> Self {
        self.0.expected = Some(f.to_string());
        self
    }
    #[inline]
    pub fn actual(self, f: impl ToString) -> Self {
        self.0.actual = Some(f.to_string());
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

#[derive(Copy, Clone, Debug)]
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
impl<'a> PartialEq<Token<'a>> for SpannedToken<'a> {
    #[inline]
    fn eq(&self, other: &Token<'a>) -> bool {
        self.kind == *other
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SeperatorParseState {
    AwaitingStart,
    AwaitingNext,
    AwaitingSeperator,
    Finished,
}
impl Default for SeperatorParseState {
    #[inline]
    fn default() -> SeperatorParseState {
        SeperatorParseState::AwaitingStart
    }
}
pub trait EndFunc<'src, 'a> {
    fn should_end(&mut self, parser: &Parser<'src, 'a>) -> bool;
    fn description(&self) -> &'static str;
}
impl<'src, 'a> EndFunc<'src, 'a> for Token<'a> {
    #[inline]
    fn should_end(&mut self, parser: &Parser<'src, 'a>) -> bool {
        match parser.peek() {
            Some(tk) => tk == *self,
            None => false
        }
    }
    #[inline]
    fn description(&self) -> &'static str {
        self.static_text().unwrap_or("ending")
    }
}
impl<'src, 'a, Func> EndFunc<'src, 'a> for Func 
    where Func: FnMut(&Parser<'src, 'a>) -> bool {
    #[inline]
    fn should_end(&mut self, parser: &Parser<'src, 'a>) -> bool {
        (*self)(parser)
    }
    #[inline]
    fn description(&self) -> &'static str {
        "ending"
    }
}
pub trait IParser<'src, 'a>: Sized {
    fn as_mut_parser(&mut self) -> &mut Parser<'src, 'a>;
    fn as_parser(&self) -> &Parser<'src, 'a>;
    #[inline]
    fn parse_terminated<'p, T, F>(
        &'p mut self,
        sep: Token<'a>,
        ending: Token<'a>,
        parse_func: F
    ) -> ParseSeperated<'p, 'src, 'a, Self, F, Token<'a>, T> 
        where F: FnMut(&mut Self) -> Result<T, ParseError> {
        ParseSeperated::new(
            self, parse_func,
            sep,
            ending,
            true,
        )
    }

}
#[derive(Debug)]
pub struct ParseSeperated<
    'p, 'src: 'p, 'a: 'p, P: IParser<'src, 'a>,
    ParseFunc, E, T
> {
    pub parser: &'p mut P,
    pub parse_func: ParseFunc,
    pub seperator: Token<'a>,
    pub end_func: E,
    pub state: SeperatorParseState,
    /// Allow an extra (redundant) separator
    /// to terminate the list of parsed items.
    ///
    /// For example: `(a, b, c,)` has a redundant comma
    /// terminating the tuple.
    pub allow_terminator: bool,
    pub marker: PhantomData<fn(&'src ()) -> T>,
}
impl<'p, 'src, 'a,
    P: IParser<'src, 'a>,
    ParseFunc: FnMut(&mut P) -> Result<T, ParseError>,
    E: EndFunc<'src, 'a>, T
> ParseSeperated<'p, 'src, 'a, P, ParseFunc, E, T> {
    #[inline]
    fn new(
        parser: &'p mut P,
        parse_func: ParseFunc,
        seperator: Token<'a>,
        end_func: E,
        allow_terminator: bool,
    ) -> Self {
        ParseSeperated {
            parser, parse_func, seperator,
            end_func, state: SeperatorParseState::AwaitingStart,
            allow_terminator, marker: PhantomData,
        }
    }
    /// The error to give if we don't encounter
    /// the expected separator
    ///
    /// For example, if we are parsing the list:
    /// `[a, b, c !` and we encounter `!`.
    /// A good message in this case would be:
    /// "Expected either ',' or ']'"
    #[cold]
    fn unexpected_seperator(&self) -> ParseError {
        self.parser.as_parser().unexpected(
            &format_args!(
                "Expected {:?} or {}",
                self.seperator.static_text().unwrap_or("<sep>"),
                self.end_func.description()
            )
        )
    }
    #[inline]
    fn maybe_end_parse(&mut self) -> bool {
        if self.end_func.should_end(self.parser.as_parser()) {
            self.state = SeperatorParseState::Finished;
            true // We want to end the parse
        } else {
            false
        }
    }
}

impl<'p, 'src, 'a,
    P: IParser<'src, 'a>,
    ParseFunc: FnMut(&mut P) -> Result<T, ParseError>,
    E: EndFunc<'src, 'a>, T
> Iterator for ParseSeperated<'p, 'src, 'a, P, ParseFunc, E, T> {
    type Item = Result<T, ParseError>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.state {
                SeperatorParseState::AwaitingStart => {
                    /*
                     * We've seen nothing yet
                     * Decide if we should end it 
                     * and abort early.
                     * This corresponds to cases:
                     * 1. [... -> starting out, no early end
                     * 2. [] -> early end
                     */
                    if self.maybe_end_parse() {
                        return None;
                    }
                    // fallthrough to parse item
                },
                SeperatorParseState::AwaitingNext => {
                    /*
                     * We've already seen a separator
                     * and are ready to parse the next item.
                     * This corresponds to case:
                     * [a, b, c -> We are right before 'c' and ready to parse it.
                     * Seeing an ending `]` would just be a plain old
                     *
                     * The only exception to this is if
                     * we allow extra terminators.
                     * We could potentially have [a, b,] in that case.
                     */
                    if self.allow_terminator && self.maybe_end_parse() {
                        return None;
                    }
                    // fallthrough to parse item
                }
                SeperatorParseState::AwaitingSeperator => {
                    let parser = self.parser.as_mut_parser();
                    match parser.peek() {
                        Some(tk) if tk == self.seperator => {
                            match parser.skip() {
                                Ok(()) => {},
                                Err(e) => return Some(Err(e))
                            };
                            self.state = SeperatorParseState::AwaitingNext;
                            continue; // Continue parsing
                        },
                        Some(_) => {
                            // We didn't see a seperator, but maybe we should end the parse
                            if self.maybe_end_parse() {
                                return None;
                            } else {
                                // fallthrough to error
                            }
                        }
                        None => {
                            // EOF -> fallthrough to error
                        }
                    };
                    return Some(Err(self.unexpected_seperator()))
                },
                SeperatorParseState::Finished => {
                    return None
                }
            }
            match (self.parse_func)(&mut *self.parser) {
                Ok(val) => {
                    self.state = SeperatorParseState::AwaitingSeperator;
                    return Some(Ok(val))
                },
                Err(e) => {
                    return Some(Err(e));
                }
            }
        }
    }
}

#[derive(educe::Educe)]
#[educe(Debug)]
pub struct Parser<'src, 'a> {
    #[educe(Debug(ignore))]
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
impl<'src, 'a> Parser<'src, 'a> {
    /// The span of the next token (same as given by peek)
    ///
    /// If this is at the EOF, gives the last token.
    #[inline]
    pub fn current_span(&self) -> Span {
        match self.buffer.back() {
            Some(tk) => tk.span,
            None => self.lexer.current_span()
        }
    }
    #[inline]
    pub fn peek_tk(&self) -> Option<SpannedToken<'a>> {
        self.buffer.back().cloned()
    }
    #[inline]
    pub fn peek(&self) -> Option<Token<'a>> {
        match self.buffer.back() {
            Some(tk) => Some(tk.kind),
            None => None
        }
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
    /// Skips over the next token without returning it
    ///
    /// This should be used in conjunction with peek.
    /// For example:
    /// ````no_run
    /// # fn result(val: i64) -> i64 {} 
    /// # fn taco() -> Result<i64, ParseError> {
    /// # let parser: Parser<'static, 'static> = todo!();
    /// match parser.peek()?.kind {
    ///     Token::Integer(val) => {
    ///         parser.skip()?;
    ///         return Ok(result(val));
    ///     }
    ///     _ => return Err(())
    /// }
    /// # }
    /// ````
    /// This is just like `pop`, but doesn't return the reuslt token.
    ///
    /// NOTE: Panics on EOF. It is the caller's responsibility to check
    /// this. This should be fine if you've already done a call to `peek`.
    #[inline]
    pub fn skip(&mut self) -> Result<(), ParseError> {
        match self.pop() {
            Ok(Some(SpannedToken { .. })) => Ok(()),
            Ok(None) => unreachable!("EOF"),
            Err(e) => Err(e)
        }
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
            self.buffer.push_front(match self.lexer.next() {
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
    #[inline]
    pub fn expect(&mut self, expected: Token<'a>)  -> Result<SpannedToken<'a>, ParseError> {
        self.expect_if(
            &expected.static_text().unwrap(),
            |actual| **actual == expected
        )
    }
    #[inline]
    pub fn expect_if(
        &mut self, expected: &dyn Display,
        func: impl FnOnce(&SpannedToken<'a>) -> bool
    ) -> Result<SpannedToken<'a>, ParseError> {
        self.expect_map(expected, |token| if func(token) { Some(*token) } else { None })
    }
    #[inline]
    pub fn expect_map<T>(
        &mut self, expected: &dyn Display,
        func: impl FnOnce(&SpannedToken<'a>) -> Option<T>
    ) -> Result<T, ParseError> {
        match self.peek_tk() {
            Some(actual) => {
                if let Some(res) = func(&actual) {
                    match self.pop()? {
                        Some(tk) => {
                            debug_assert_eq!(tk, actual.kind);
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
            ParseErrorKind::UnexpectedEof
        } else {
            ParseErrorKind::UnexpectedToken
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
        self.expect_map(&"an identifier", |token| match token.kind {
            Token::Ident(ident) => Some(ident),
            _ => None
        })
    }
}