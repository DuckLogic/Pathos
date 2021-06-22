use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Deref;

use crate::alloc::{Allocator};
use crate::ast::Span;
use crate::lexer::{LexError, PythonLexer, Token};
use crate::parse::errors::{ParseError, ParseErrorKind, LineNumberTracker};

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
pub struct EndFunc<'src, 'a, F, D: Display> {
    func: F,
    description: D,
    marker: PhantomData<fn(&'src (), &'a ())>
}
impl<'src, 'a, F: FnMut(&Parser<'src, 'a>) -> bool, D: Display> EndFunc<'src, 'a, F, D> {
    #[inline]
    pub fn new(description: D, func: F) -> Self {
        EndFunc { func, description, marker: PhantomData }
    }
}
impl<'src, 'a, F, D> EndPredicate<'src, 'a> for EndFunc<'src, 'a, F, D> where F: FnMut(&Parser<'src, 'a>) -> bool, D: Display {
    #[inline]
    fn should_end(&mut self, parser: &Parser<'src, 'a>) -> bool {
        (self.func)(parser)
    }
    type Desc = D;
    #[inline]
    fn description(&self) -> &D {
        &self.description
    }
}
pub trait EndPredicate<'src, 'a> {
    fn should_end(&mut self, parser: &Parser<'src, 'a>) -> bool;
    type Desc: Display;
    fn description(&self) -> &Self::Desc;
}
impl<'src, 'a> EndPredicate<'src, 'a> for Token<'a> {
    #[inline]
    fn should_end(&mut self, parser: &Parser<'src, 'a>) -> bool {
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
pub trait IParser<'src, 'a>: Sized + Debug {
    fn as_mut_parser(&mut self) -> &mut Parser<'src, 'a>;
    fn as_parser(&self) -> &Parser<'src, 'a>;
    #[inline]
    fn parse_seperated<'p, T, F>(
        &'p mut self,
        sep: Token<'a>,
        ending: Token<'a>,
        parse_func: F,
        config: ParseSeperatedConfig
    ) -> ParseSeperated<'p, 'src, 'a, Self, F, Token<'a>, T>
        where F: FnMut(&mut Self) -> Result<T, ParseError> {
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
    'p, 'src: 'p, 'a: 'p,
    P: IParser<'src, 'a>,
    ParseFunc, E, T
> {
    pub parser: &'p mut P,
    pub parse_func: ParseFunc,
    pub separator: Token<'a>,
    pub end_func: E,
    pub state: SeparatorParseState,
    pub marker: PhantomData<fn(&'src ()) -> T>,
    pub config: ParseSeperatedConfig
}
impl<'p, 'src, 'a,
    P: IParser<'src, 'a>,
    ParseFunc: FnMut(&mut P) -> Result<T, ParseError>,
    E: EndPredicate<'src, 'a>, T
> ParseSeperated<'p, 'src, 'a, P, ParseFunc, E, T> {
    #[inline]
    pub fn new(
        parser: &'p mut P,
        parse_func: ParseFunc,
        separator: Token<'a>,
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
                self.separator.static_text().unwrap_or("<sep>"),
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
    P: IParser<'src, 'a>,
    ParseFunc: FnMut(&mut P) -> Result<T, ParseError>,
    E: EndPredicate<'src, 'a>, T
> Iterator for ParseSeperated<'p, 'src, 'a, P, ParseFunc, E, T> {
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
                            Some(Token::Newline) => {
                                // Implicitly consume line and fetch more input
                                match parser.skip() {
                                    Ok(_) => {},
                                    Err(e) => return Some(Err(e))
                                };
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
                        match parser.skip() {
                            Ok(_) => {},
                            Err(e) => return Some(Err(e))
                        };
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
pub struct Parser<'src, 'a> {
    /// The buffer of tokens, always containing at least a single line of tokens.
    ///
    /// The end of the buffer corresponds to one of two things:
    /// 1. The end of the logical line.
    /// 2. The end of the file.
    ///
    /// The buffering deals only in logical lines, not physical lines.
    buffer: Vec<SpannedToken<'a>>,
    /// The current position within the buffer
    ///
    /// Once this reaches the buffer's length,
    /// the buffer is empty and a new line needs to be fetched.
    current_index: usize,
    /// If the lexer has reached the end of the file,
    /// and there is no point asking for more.
    eof: bool,
    lexer: PythonLexer<'src, 'a>,
}

impl<'src, 'a> Parser<'src, 'a> {
    /// A limited form of look behind, which only works for the current line
    pub(crate) fn look_behind(&self, amount: usize) -> Option<Token<'a>> {
        assert!(amount >= 1);
        self.current_index.checked_sub(amount)
            .and_then(|index| self.buffer.get(index))
            .map(|tk| tk.kind)
    }
}

impl<'src, 'a> Parser<'src, 'a> {
    pub fn new(_arena: &'a Allocator, lexer: PythonLexer<'src, 'a>) -> Result<Self, ParseError> {
        let mut res = Parser {
            lexer, buffer: Vec::with_capacity(32),
            eof: false,
            current_index: 0,
        };
        res.lexer.line_number_tracker = Some(LineNumberTracker::new());
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
    pub fn peek_tk(&self) -> Option<SpannedToken<'a>> {
        self.buffer.get(self.current_index).cloned()
    }
    #[inline]
    pub fn peek(&self) -> Option<Token<'a>> {
        match self.buffer.get(self.current_index) {
            Some(tk) => Some(tk.kind),
            None => None
        }
    }
    /// Return if the next token is either `Newline` or the EOF
    #[inline]
    pub fn is_newline(&self) -> bool {
        self.peek() == Some(Token::Newline) || self.is_end_of_file()
    }
    fn assert_empty(&mut self) -> bool {
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
    ///
    /// This doesn't necessarily advance the lexer past newlines,
    /// and may return `None` if the end of line is encountered
    #[inline]
    pub fn look_ahead(&self, amount: usize) -> Result<Option<SpannedToken<'a>>, ParseError> {
        // TODO: Change signature to reflect the fact this is infallible
        Ok(self.buffer.get(self.current_index + amount).cloned())
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
    pub fn skip(&mut self) -> Result<Token<'a>, ParseError> {
        match self.pop() {
            Ok(Some(SpannedToken { kind, .. })) => Ok(kind),
            Ok(None) => unreachable!("EOL/EOF"),
            Err(e) => Err(e)
        }
    }
    /// Pop a token, advancing the position in the internal buffer.
    ///
    /// Return `None` if already at the end of the line.
    #[inline]
    pub fn pop(&mut self) -> Result<Option<SpannedToken<'a>>, ParseError> {
        // TODO: Update signature to reflect the fact this is now infallible
        match self.buffer.get(self.current_index) {
            Some(&tk) => {
                self.current_index += 1;
                Ok(Some(tk))
            },
            None => Ok(None)
        }
    }
    /// Clear the internal buffer, and reset the parser for use with parsing a new line
    pub fn reset_buffer(&mut self) -> Result<(), ParseError> {
        self.buffer.clear();
        self.current_index = 0;
        self.next_line()?; // We always have at least one line (unless we have an error)
        Ok(())
    }
    /// Give the length of input remaining in the buffer
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.current_index
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
            let tk = match self.lexer.next() {
                Ok(Some(val)) => SpannedToken { span: lexer_span, kind: val },
                Ok(None) => {
                    self.eof = true;
                    break
                },
                Err(cause) => {
                    let kind = match cause {
                        LexError::AllocFailed => {
                            // TODO: Handle OOM without boxing errors.....
                            ParseErrorKind::AllocationFailed
                        },
                        LexError::InvalidToken => ParseErrorKind::InvalidToken,
                        LexError::InvalidString(cause) => ParseErrorKind::InvalidString(cause),
                    };
                    return Err(ParseError::builder(lexer_span, kind).build());
                },
            };
            self.buffer.push(tk);
            count += 1;
            if tk.kind == Token::Newline { break }
        }
        if count == 0 && self.eof {
            Ok(None)
        } else {
            Ok(Some(count))
        }
    }
    /// The line number tracker, to give more detailed errors
    #[inline]
    pub fn line_number_tracker(&self) -> &LineNumberTracker {
        self.lexer.line_number_tracker.as_ref().unwrap()
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
    pub fn expect(&mut self, expected: Token<'a>)  -> Result<SpannedToken<'a>, ParseError> {
        self.expect_if(
            &expected.static_text().unwrap(),
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
    pub fn into_lexer(self) -> PythonLexer<'src, 'a> {
        self.lexer
    }
}