use std::backtrace::Backtrace;
use std::error::Error as StdError;
use std::fmt::{self, Display, Formatter, Debug};


use crate::alloc::AllocError;
use crate::ast::{Span, };
use crate::lexer::LexerError;
use crate::errors::tracker::{DetailedSpan, LineTracker};

#[cfg(feature = "fancy-errors")]
pub mod fancy;
pub mod tracker;

pub trait SpannedError: StdError {
    fn span(&self) -> ErrorSpan;
}
impl SpannedError for ParseError {
    #[inline]
    fn span(&self) -> ErrorSpan {
        match self.0.span {
            InternalSpan::AllocationFailed => ErrorSpan::AllocationFailed,
            InternalSpan::Detailed { original: span, .. } |
            InternalSpan::Regular(span) => ErrorSpan::Span(span)
        }
    }
}

/// Either the [Span] of an error, or a valid reason one couldn't be created.
#[derive(Debug, Clone, Copy)]
pub enum ErrorSpan {
    /// Indicates that the error is actually caused by an allocation failure,
    /// so no error is present
    AllocationFailed,
    Span(Span)
}
impl From<Span> for ErrorSpan {
    fn from(s: Span) -> Self {
        ErrorSpan::Span(s)
    }
}
impl ErrorSpan {
    #[track_caller]
    pub fn unwrap(&self) -> Span {
        match *self {
            ErrorSpan::AllocationFailed => panic!("Missing span: Allocation failed"),
            ErrorSpan::Span(span) => span
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ParseErrorKind {
    Lexer(Box<dyn LexerError>),
    InvalidExpression,
    AllocationFailed,
    UnexpectedEof,
    UnexpectedEol,
    UnexpectedToken,
}
#[derive(Copy, Clone, Debug)]
pub(crate) enum InternalSpan {
    AllocationFailed,
    Detailed {
        original: Span,
        detailed: DetailedSpan
    },
    Regular(Span)
}
#[derive(Debug)]
struct ParseErrorInner {
    /// The span of the source location
    ///
    /// There is only one instance where this can be `None`:
    /// 1. Out of memory errors
    span: InternalSpan,
    expected: Option<String>,
    actual: Option<String>,
    kind: ParseErrorKind,
    backtrace: Backtrace
}

pub struct ParseError(Box<ParseErrorInner>);
impl ParseError {
    pub fn with_line_numbers(mut self, line_number_tracker: &LineTracker) -> Self {
        self.0.span = match self.0.span {
            InternalSpan::AllocationFailed => InternalSpan::AllocationFailed,
            InternalSpan::Detailed { original, detailed } => InternalSpan::Detailed { original, detailed },
            InternalSpan::Regular(span) => InternalSpan::Detailed {
                original: span,
                detailed: line_number_tracker.resolve_span(span)
            }
        };
        self
    }
    /// Give additional context on the type of item that was "expected"
    pub fn with_expected_msg<T: ToString>(mut self, msg: T) -> Self {
        self.0.expected = Some(msg.to_string());
        self
    }
    #[inline]
    pub fn expected_msg(&self) -> Option<&str> {
        self.0.expected.as_deref()
    }
    pub fn with_actual_msg<T: ToString>(mut self, msg: T) -> Self {
        self.0.actual = Some(msg.to_string());
        self
    }
    #[inline]
    pub fn builder(span: Span, kind: ParseErrorKind) -> ParseErrorBuilder {
        ParseErrorBuilder(ParseErrorInner {
            span: InternalSpan::Regular(span), kind,
            expected: None, actual: None,
            backtrace: Backtrace::disabled() // NOTE: Actual capture comes later
        })
    }
}
impl From<AllocError> for ParseError {
    #[cold]
    #[track_caller]
    fn from(_cause: AllocError) -> Self {
        /*
         * TODO: Handle this without allocating.
         * Its bad to allocate memory for the error value
         * even though we have an 'out of memory' condition.
         * Realistically though, we'll probably only encounter
         * OOM if the counter hits the internal limit.
         */
        ParseError(Box::new(ParseErrorInner {
            span: InternalSpan::AllocationFailed,
            expected: None,
            actual: None,
            kind: ParseErrorKind::AllocationFailed,
            backtrace: Backtrace::capture()
        }))
    }
}
impl Debug for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ParseError");
        match self.0.span {
            InternalSpan::Detailed { detailed, .. } => {
                debug.field("span", &format_args!("{}", detailed));
            },
            InternalSpan::Regular(span) => {
                debug.field("span", &format_args!("{}", span));
            }
            InternalSpan::AllocationFailed => {}
        }
        debug.field("expected", &self.0.expected)
            .field("actual", &self.0.actual)
            .field("kind", &self.0.kind)
            .field("backtrace", &format_args!("{}", self.0.backtrace))
            .finish()
    }
}
impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0.kind {
            ParseErrorKind::Lexer(ref cause) => {
                write!(f, "{}", cause)?;
            },
            ParseErrorKind::InvalidExpression => {
                f.write_str("Invalid expression")?;
            },
            ParseErrorKind::AllocationFailed => {
                f.write_str("Allocation failed")?;
            }
            ParseErrorKind::UnexpectedEof => {
                f.write_str("Unexpected EOF")?;
            },
            ParseErrorKind::UnexpectedEol => {
                f.write_str("Unexpected end of line")?;
            }
            ParseErrorKind::UnexpectedToken => {
                f.write_str("Unexpected token")?;
            }
        }
        match (&self.0.expected, &self.0.actual) {
            (Some(ref expected), Some(ref actual)) => {
                write!(f, ": Expected {}, but got {}", expected, actual)?;
            },
            (Some(ref expected), None) => {
                write!(f, ": Expected {}", expected)?;
            },
            (None, Some(ref actual)) => {
                write!(f, ": Got {}", actual)?;
            },
            (None, None) => {}
        }
        match self.0.span {
            InternalSpan::Regular(span) => {
                write!(f, " @ {}", span)?;
            },
            InternalSpan::Detailed { detailed, .. } => {
                write!(f, " @ {}", detailed)?;
            },
            InternalSpan::AllocationFailed => {}
        }
        Ok(())
    }
}

impl StdError for ParseError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self.0.kind {
            ParseErrorKind::Lexer(ref cause) => Some(cause.upcast()),
            _ => None
        }
    }

    #[inline]
    fn backtrace(&self) -> Option<&Backtrace> {
        Some(&self.0.backtrace)
    }

}

pub struct ParseErrorBuilder(ParseErrorInner);

impl ParseErrorBuilder {
    #[inline]
    pub fn expected(mut self, f: impl ToString) -> Self {
        self.0.expected = Some(f.to_string());
        self
    }
    #[inline]
    pub fn actual(mut self, f: impl ToString) -> Self {
        self.0.actual = Some(f.to_string());
        self
    }
    #[inline]
    pub fn build(mut self) -> ParseError {
        ParseError(Box::new(ParseErrorInner {
            span: self.0.span,
            kind: self.0.kind,
            expected: self.0.expected.take(),
            actual: self.0.actual.take(),
            backtrace: Backtrace::capture()
        }))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::tracker::LineTracker;
    use std::cell::Cell;

    #[test]
    fn single_line() {
        // Test ASCII
        assert_eq!(LineTracker::from_text("").lines[0].num_bytes(), 0);
        let mut tracker = LineTracker::from_text("All cows eat grass");
        assert_eq!(tracker.lines[0], LineCache {
            char_offset: 0,
            start: 0,
            end: 18,
            character_boundaries: None
        });
        assert_eq!(tracker.resolve_position(Position(4)), DetailedLocation {
            line_number: LineNumber::ONE,
            column: 4
        });
        // Test unicode
        tracker = LineTracker::from_text("Unicode: \u{E2}ll \u{14D}ows eat gr\u{1CE}ss");
        assert_eq!(tracker.lines[0].start, 0);
        let difference_between_chars_and_bytes = Cell::new(0);
        let verify_char_position = |char_index: u64| {
            assert_eq!(
                tracker.resolve_position(Position(char_index + difference_between_chars_and_bytes.get())),
                DetailedLocation {
                    line_number: LineNumber::ONE,
                    column: char_index
                }
            );
        };
        // 3rd char is still ASCII
        assert_eq!(difference_between_chars_and_bytes.get(), 0);
        verify_char_position(3);
        // 9th character is unicode, but it *starts* at byte-index 9 (because everything before it is ASCII)
        verify_char_position(9);
        // Character 9 is two bytes, offsetting all future characters by one
        difference_between_chars_and_bytes.update(|i| i + 1);
        verify_char_position(10);
        verify_char_position(11);
        // Character 13 is also unicode and its two bytes
        verify_char_position(13);
        difference_between_chars_and_bytes.update(|i| i + 1);
        verify_char_position(14);
        // Character 24 is also unicode and its two bytes
        verify_char_position(24);
        difference_between_chars_and_bytes.update(|i| i + 1);
        verify_char_position(25);
        verify_char_position(26);
    }
    fn verify(text: &str) {
        let tracker = LineTracker::from_text(text);
        let mut last_line = 0;
        for (line_index, line) in text.split('\n').enumerate() {
            let line_number = LineNumber::from_index(line_index);
            for (char_offset, (offset, _)) in line.char_indices().enumerate() {
                assert_eq!(tracker.resolve_position((last_line + offset).into()), DetailedLocation {
                    column: char_offset as u64,
                    line_number
                }, "Mismatched locations for {} at {} in line #{}", char_offset, offset, line_number);
            }
            assert_eq!(tracker.resolve_position(Position(last_line as u64 + line.len() as u64)), DetailedLocation {
                line_number, column: line.chars().count() as u64
            });
            last_line += line.len() + 1;
        }
    }
    #[test]
    fn test_multiline() {
        verify("\nAll cows eat grass");
        verify("All cows\n eat grass");
        verify("All cows\n eat grass\n");
        verify("Unicode: \u{E2}ll \u{14D}\nows eat gr\u{1CE}ss");
        verify("Unicode: \u{E2}ll \u{14D}\nows eat gr\u{1CE}ss\n");
        verify("Unicode: \u{E2}ll\n \u{14D}ows eat gr\u{1CE}ss\n");
    }
}
