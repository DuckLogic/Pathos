use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::alloc::{Allocator, AllocError};
use crate::ast::Span;
use crate::lexer::{Lexer, Token, SpannedToken, LexerError};
use crate::errors::{ParseError, ParseErrorKind, tracker::LineTracker, SpannedError};

#[derive(Copy, Clone, Debug)]
pub enum SeparatorParseState {
    AwaitingStart,
    AwaitingNext,
    AwaitingSeparator,
    Finished,
}
impl Default for SeparatorParseState {
    #[inline]
    fn default() -> SeparatorParseState {
        SeparatorParseState::AwaitingStart
    }
}
pub struct EndFunc<'a, F, L: Lexer<'a>, D: Display> {
    func: F,
    description: D,
    marker: PhantomData<fn(&'a (), L)>
}
impl<'src, 'a, F: FnMut(&Parser<'a, L>) -> bool, L: Lexer<'a>, D: Display> EndFunc<'a, F, L, D> {
    #[inline]
    pub fn new(description: D, func: F) -> Self {
        EndFunc { func, description, marker: PhantomData }
    }
}
impl<'a, F, D, L: Lexer<'a>> EndPredicate<'a, L::Token, L> for EndFunc<'a, F, L, D>
    where F: FnMut(&Parser<'a, L>) -> bool, D: Display {
    #[inline]
    fn should_end(&mut self, parser: &Parser<'a, L>) -> bool {
        (self.func)(parser)
    }
    type Desc = D;
    #[inline]
    fn description(&self) -> &D {
        &self.description
    }
}
pub trait EndPredicate<'a, T: Token<'a>, L: Lexer<'a, Token=T>> {
    fn should_end(&mut self, parser: &Parser<'a, L>) -> bool;
    type Desc: Display;
    fn description(&self) -> &Self::Desc;
}
impl<'a, T: Token<'a>, L: Lexer<'a, Token=T>> EndPredicate<'a, T, L> for T {
    #[inline]
    fn should_end(&mut self, parser: &Parser<'a, L>) -> bool {
        match parser.peek() {
            Some(tk) => tk == *self,
            None => false
        }
    }
    type Desc = Self;
    #[inline]
    fn description(&self) -> &Self {
        self
    }
}
pub trait IParser<'a>: Sized + Debug {
    type Token: Token<'a>;
    type Lexer: Lexer<'a, Token=Self::Token>;
    fn as_mut_parser(&mut self) -> &mut Parser<'a, Self::Lexer>;
    fn as_parser(&self) -> &Parser<'a, Self::Lexer>;
    #[inline]
    fn parse_seperated<'p, T, F, E>(
        &'p mut self,
        sep: Self::Token,
        ending: E,
        parse_func: F,
        config: ParseSeperatedConfig
    ) -> ParseSeperated<'p, 'a, Self, F, E, T>
        where F: FnMut(&mut Self) -> Result<T, ParseError>, E: EndPredicate<'a, Self::Token, Self::Lexer> {
        ParseSeperated::new(
            self, parse_func,
            sep,
            ending,
            config
        )
    }
}
/// Configuration for [ParseSeperated]
#[derive(Copy, Clone, Debug)]
pub struct ParseSeperatedConfig {
    /// Allow an extra (redundant) separator
    /// to terminate the list of parsed items.
    ///
    /// For example: `(a, b, c,)` has a redundant comma
    /// terminating the tuple.
    pub allow_terminator: bool,
    /// Allow input to span multiple lines
    pub allow_multi_line: bool
}
#[derive(Debug)]
pub struct ParseSeperated<
    'p, 'a: 'p,
    P: IParser<'a>,
    ParseFunc, E, T
> {
    pub parser: &'p mut P,
    pub parse_func: ParseFunc,
    pub separator: P::Token,
    pub end_func: E,
    pub state: SeparatorParseState,
    pub marker: PhantomData<fn() -> T>,
    pub config: ParseSeperatedConfig,
}
impl<'p, 'a,
    P: IParser<'a>,
    ParseFunc: FnMut(&mut P) -> Result<T, ParseError>,
    E: EndPredicate<'a, P::Token, P::Lexer>, T
> ParseSeperated<'p, 'a, P, ParseFunc, E, T> {
    #[inline]
    pub fn new(
        parser: &'p mut P,
        parse_func: ParseFunc,
        separator: P::Token,
        end_func: E,
        config: ParseSeperatedConfig
    ) -> Self {
        ParseSeperated {
            parser, parse_func,
            separator,
            end_func, state: SeparatorParseState::AwaitingStart,
            config, marker: PhantomData,
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
    fn unexpected_separator(&self) -> ParseError {
        self.parser.as_parser().unexpected(
            &format_args!(
                "Expected {:?} or ending {}",
                self.separator,
                self.end_func.description()
            )
        )
    }
    #[inline]
    fn maybe_end_parse(&mut self) -> bool {
        if self.end_func.should_end(self.parser.as_parser()) {
            self.state = SeparatorParseState::Finished;
            true // We want to end the parse
        } else {
            false
        }
    }
}

impl<'p, 'src, 'a: 'p,
    P: IParser<'a>,
    ParseFunc: FnMut(&mut P) -> Result<T, ParseError>,
    E: EndPredicate<'a, P::Token, P::Lexer>, T
> Iterator for ParseSeperated<'p, 'a, P, ParseFunc, E, T> {
    type Item = Result<T, ParseError>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.state {
                SeparatorParseState::AwaitingStart => {
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
                SeparatorParseState::AwaitingNext => {
                    if self.config.allow_multi_line {
                        /*
                         * If we're allowing multi-line input,
                         * and we've already seen a separator,
                         * implicitly fetch a newline if we need to.
                         * This corresponds to the implicit line joining logic described here:
                         * https://docs.python.org/3.11/reference/lexical_analysis.html#implicit-line-joining
                         * To quote, "Expressions in parentheses, square brackets or curly braces
                         * can be split over more than one physical line without using backslashes"
                         */
                        let parser = self.parser.as_mut_parser();
                        let should_eat_more = match parser.peek() {
                            Some(tk) if tk.is_newline() => {
                                // Implicitly consume line and fetch more input
                                parser.skip();
                                if parser.is_empty() {
                                    true
                                } else {
                                    continue // more input -> continue outer loop
                                }
                            },
                            None => {
                                true
                            },
                            _ => false // fallthrough to regular parse
                        };
                        if should_eat_more {
                            // Implicitly fetch more input
                            match parser.next_line() {
                                Ok(Some(_)) => {
                                    // More input
                                    continue
                                },
                                Ok(None) => {
                                    // fallthrough to regular parse, which handles EOF condition
                                },
                                Err(cause) => return Some(Err(cause))
                            }
                        }
                    }
                    /*
                     * We've already seen a separator
                     * and are ready to parse the next item.
                     * This corresponds to case:
                     * [a, b, c -> We are right after the comma, right before 'c', and ready to parse
                     * the next item item.
                     * If `!self.allow_terminator`, we dont allow trailing commas
                     * so seeing an ending `]` would be error.
                     */
                    if self.config.allow_terminator && self.maybe_end_parse() {
                        // Redundant comma
                        return None;
                    }
                    // fallthrough to parse item
                }
                SeparatorParseState::AwaitingSeparator => {
                    let parser = self.parser.as_mut_parser();
                    if parser.peek() == Some(self.separator) {
                        parser.skip();
                        self.state = SeparatorParseState::AwaitingNext;
                        continue; // Continue parsing
                    } else {
                        // We didn't see a separator, but maybe we should end the parse
                        if self.maybe_end_parse() {
                            return None;
                        } else {
                            // fallthrough to error
                        }
                    }
                    return Some(Err(self.unexpected_separator()))
                },
                SeparatorParseState::Finished => {
                    return None
                }
            }
            match (self.parse_func)(&mut *self.parser) {
                Ok(val) => {
                    self.state = SeparatorParseState::AwaitingSeparator;
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
pub struct Parser<'a, L: Lexer<'a>> {
    /// The buffer of tokens, always containing at least a single line of tokens.
    ///
    /// The end of the buffer corresponds to one of two things:
    /// 1. The end of the logical line.
    /// 2. The end of the file.
    ///
    /// The buffering deals only in logical lines, not physical lines.
    buffer: Vec<SpannedToken<'a, L::Token>>,
    /// The current position within the buffer
    ///
    /// Once this reaches the buffer's length,
    /// the buffer is empty and a new line needs to be fetched.
    current_index: usize,
    /// If the lexer has reached the end of the file,
    /// and there is no point asking for more.
    eof: bool,
    lexer: L,
    marker: PhantomData<&'a ()>,
    // TODO: Encapsulate
    pub line_tracker: LineTracker,
}

impl<'a, L: Lexer<'a>> Parser<'a, L> {
    /// A limited form of look behind, which only works for the current line
    pub fn look_behind(&self, amount: usize) -> Option<L::Token> {
        assert!(amount >= 1);
        self.current_index.checked_sub(amount)
            .and_then(|index| self.buffer.get(index))
            .map(|tk| tk.kind)
    }
    pub fn new(_arena: &'a Allocator, lexer: L) -> Result<Self, ParseError> {
        let line_tracker = LineTracker::from_text(lexer.original_text());
        let mut res = Parser {
            lexer, buffer: Vec::with_capacity(32),
            eof: false,
            current_index: 0,
            line_tracker,
            marker: PhantomData
        };
        res.next_line()?;
        Ok(res)
    }
    /// The span of the next token (same as given by peek)
    ///
    /// If this is at the end of line, this gives the last token.
    #[inline]
    pub fn current_span(&self) -> Span {
        match self.peek_tk() {
            Some(tk) => tk.span,
            None => self.lexer.current_span()
        }
    }
    #[inline]
    pub fn peek_tk(&self) -> Option<SpannedToken<'a, L::Token>> {
        self.buffer.get(self.current_index).cloned()
    }
    #[inline]
    pub fn peek(&self) -> Option<L::Token> {
        self.buffer.get(self.current_index).map(|tk| tk.kind)
    }
    /// Return if the next token is either `Newline` or the EOF
    #[inline]
    pub fn is_newline(&self) -> bool {
        self.peek().map_or(false, |tk| tk.is_newline()) || self.is_end_of_file()
    }
    /// Look ahead the specified number of tokens
    ///
    /// If `amount == 0`, then this is equivalent
    /// to calling [Parser::peek]
    ///
    /// This doesn't necessarily advance the lexer past newlines,
    /// and may return `None` if the end of line is encountered
    #[inline]
    pub fn look_ahead(&self, amount: usize) -> Option<L::Token> {
        self.buffer.get(self.current_index + amount).map(|tk| tk.kind)
    }
    /// Skips over the next token without returning it
    ///
    /// This should be used in conjunction with peek.
    /// For example:
    /// ````ignore
    /// match parser.peek()?.kind {
    ///     Token::Integer(val) => {
    ///         parser.skip()?;
    ///         return Ok(result(val));
    ///     }
    ///     _ => return Err(())
    /// }
    /// ````
    /// This is just like `pop`, but doesn't return the result token.
    ///
    /// NOTE: Panics on end of line. It is the caller's responsibility to check
    /// this. This should be fine if you've already done a call to `peek`.
    #[inline]
    #[track_caller]
    pub fn skip(&mut self) -> SpannedToken<'a, L::Token> {
        self.pop().expect("EOL/EOF")
    }
    /// Pop a token, advancing the position in the internal buffer.
    ///
    /// Return `None` if already at the end of the line.
    #[inline]
    #[must_use = "Did you mean skip()? pop() does nothing if the parser is empty..."]
    pub fn pop(&mut self) -> Option<SpannedToken<'a, L::Token>> {
        match self.buffer.get(self.current_index) {
            Some(&tk) => {
                self.current_index += 1;
                Some(tk)
            },
            None => None
        }
    }
    /// Advance the tokenizer, and fill the buffer with the next (logical) line of input
    ///
    /// NOTE: This method only cares about *logical* lines. A single logical line can
    /// span across multiple physical lines.
    ///
    /// Returns the number of tokens consumed, or `None` if the EOF is reached
    pub fn next_line(&mut self) -> Result<Option<usize>, ParseError> {
        if self.eof {
            return Ok(None);
        }
        let mut count = 0;
        loop {
            let lexer_span = self.lexer.current_span();
            let tk = match self.lexer.try_next() {
                Ok(Some(val)) => SpannedToken { span: lexer_span, kind: val, marker: PhantomData },
                Ok(None) => {
                    self.eof = true;
                    break
                },
                Err(cause) => {
                    let span = cause.span();
                    if cause.cast_alloc_failed().is_some() {
                        return Err(ParseError::from(AllocError))
                    };
                    return Err(ParseError::builder(span.unwrap(), ParseErrorKind::Lexer(Box::new(cause)))
                        .build());
                },
            };
            self.buffer.push(tk);
            count += 1;
            if tk.kind.is_newline() { break }
        }
        if count == 0 && self.eof {
            Ok(None)
        } else {
            Ok(Some(count))
        }
    }
    /// Check if the internal buffer is empty.
    ///
    /// This can only be true if there is an end of file,
    /// or if there is an end of line.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current_index == self.buffer.len()
    }
    /// Check if the parser reached EOF
    ///
    /// This can be used to disambiguate between end of line (EOL) and end of file (EOF)
    #[inline]
    pub fn is_end_of_file(&self) -> bool {
        self.current_index == self.buffer.len() && self.eof
    }
    //
    // Utilities
    //
    pub fn expect(&mut self, expected: L::Token) -> Result<SpannedToken<'a, L::Token>, ParseError> {
        self.expect_if(
            &format_args!("{:?}", expected),
            |actual| **actual == expected
        )
    }
    /// Expect that the parser has completely finished parsing its input
    ///
    /// This is typically done at the end of a top-level (user-visible) parse function.
    ///
    /// In order to give descriptive error messages,
    /// this may need to advance the lexer and read a new line
    pub fn expect_end_of_input(&mut self) -> Result<(), ParseError> {
        if self.is_end_of_file() {
            return Ok(())
        }
        if self.is_empty() {
            self.next_line()?;
        }
        match self.peek() {
            None => unreachable!(),
            Some(tk) => {
                Err(ParseError::builder(self.current_span(), ParseErrorKind::UnexpectedToken)
                    .expected("end of input")
                    .actual(tk)
                    .build())
            }
        }
    }
    #[inline]
    pub fn expect_if(
        &mut self, expected: &dyn Display,
        func: impl FnOnce(&SpannedToken<'a, L::Token>) -> bool
    ) -> Result<SpannedToken<'a, L::Token>, ParseError> {
        self.expect_map(expected, |token| if func(token) { Some(*token) } else { None })
    }
    #[inline]
    pub fn expect_map<T>(
        &mut self, expected: &dyn Display,
        func: impl FnOnce(&SpannedToken<'a, L::Token>) -> Option<T>
    ) -> Result<T, ParseError> {
        if let Some(peeked) = self.peek_tk() {
            if let Some(res) = func(&peeked) {
                let actual = self.skip();
                debug_assert_eq!(peeked.kind, actual.kind);
                debug_assert_eq!(peeked.span, actual.span);
                return Ok(res);
            }
        }
        // fallthrough to error
        Err(self.unexpected(expected))
    }
    #[cold]
    pub fn unexpected(&self, expected: &dyn Display) -> ParseError {
        let kind = if self.is_end_of_file() {
            ParseErrorKind::UnexpectedEof
        } else if self.is_empty() {
            ParseErrorKind::UnexpectedEol
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
    #[inline]
    pub fn into_lexer(self) -> L {
        self.lexer
    }
}