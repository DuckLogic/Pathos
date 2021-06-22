use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt::{Display, Formatter, Debug};
use std::fmt;

use crate::alloc::AllocError;
use crate::ast::Span;
use crate::lexer::StringError;

/// A detailed location in the source
#[derive(Copy, Clone)]
pub struct DetailedLocation {
    /// The one-based line number
    pub line_number: u64,
    /// The zero-based offset of the character within the line
    pub column: u64,
}
impl Display for DetailedLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line_number, self.column)
    }
}
/// A more detailed version of [Span]
///
/// Contains both line number and column information
#[derive(Copy, Clone)]
pub struct DetailedSpan {
    pub start: DetailedLocation,
    pub end: DetailedLocation
}
#[derive(Debug)]
pub struct LineNumberTracker {
    line_starts: Vec<u64>,
}
impl LineNumberTracker {
    pub fn new() -> LineNumberTracker {
        LineNumberTracker { line_starts: vec![0] }
    }
    pub fn reset(&mut self)  {
        self.line_starts.clear();
        self.line_starts.push(0);
    }
    /// Resolve a byte-based index into a detailed location
    pub fn resolve_location(&self, index: u64) -> DetailedLocation {
        let line_index = match self.line_starts.binary_search(&index) {
            Ok(index) => {
                // The index is *exactly* the start of a line
                index
            },
            Err(insertion_index) => {
               /*
                * The index comes after the start of a line.
                * The 'insertion_index' is where the index could be inserted to maintain sorted order.
                * For example, with line starts [0, 2, 7, 14, 18]
                * The value '8' would give the index '3'.
                * Subtract one from the insertion_index to get the value before (the line's start location)
                * NOTE: We know the subtraction cannot overflow,
                * because index > 0 (otherwise we would've been Ok)
                */
                insertion_index.checked_sub(1).unwrap()
            }
        };
        let line_starting_index = self.line_starts[line_index];
        let line_number = line_index + 1;
        DetailedLocation {
            line_number: line_number as u64,
            column: index - line_starting_index
        }
    }
    pub fn resolve_span(&self, span: Span) -> DetailedSpan {
        DetailedSpan {
            start: self.resolve_location(span.start),
            end: self.resolve_location(span.end)
        }
    }
    /// Mark an index as the start of a newline
    ///
    /// This must always be done in increasing order.
    ///
    /// This marks the index
    ///
    /// NOTE: `0` is always (implicitly)
    /// the start of a newline and does not need to be marked explicitly
    pub fn mark_line_start(&mut self, line_start: u64) {
        let last_line_start = *self.line_starts.last().unwrap();
        assert!(line_start > last_line_start, "Next line start {} must come after last line start {}", line_start, last_line_start);
        self.line_starts.push(line_start);
    }
}
impl Debug for DetailedSpan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
impl Display for DetailedSpan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.start.line_number == self.end.line_number {
            write!(f, "{}:{}..{}", self.start.line_number, self.start.column, self.end.column)
        } else {
            write!(f, "{}..{}", self.start, self.end)
        }
    }
}
#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    InvalidToken,
    AllocationFailed,
    UnexpectedEof,
    UnexpectedEol,
    UnexpectedToken,
    InvalidString(StringError),
}
#[derive(Copy, Clone, Debug)]
pub enum MaybeSpan {
    Missing,
    Detailed(DetailedSpan),
    Regular(Span)
}
#[derive(Debug)]
struct ParseErrorInner {
    /// The span of the source location
    ///
    /// There are only two instances where this can be `None`:
    /// 1. Out of memory errors
    /// 2. A ParseVisitError
    span: MaybeSpan,
    expected: Option<String>,
    actual: Option<String>,
    kind: ParseErrorKind,
    backtrace: Backtrace
}

pub struct ParseError(Box<ParseErrorInner>);
impl ParseError {
    pub fn with_line_numbers(mut self, line_number_tracker: &LineNumberTracker) -> Self {
        self.0.span = match self.0.span {
            MaybeSpan::Missing => MaybeSpan::Missing,
            MaybeSpan::Detailed(already_detailed) => MaybeSpan::Detailed(already_detailed),
            MaybeSpan::Regular(span) => MaybeSpan::Detailed(line_number_tracker.resolve_span(span))
        };
        self
    }
    /// Give additional context on the type of item that was "expected"
    #[cold]
    pub fn with_expected_msg<T: ToString>(mut self, msg: T) -> Self {
        self.0.expected = Some(msg.to_string());
        self
    }
    #[inline]
    pub fn builder(span: Span, kind: ParseErrorKind) -> ParseErrorBuilder {
        ParseErrorBuilder(ParseErrorInner {
            span: MaybeSpan::Regular(span), kind,
            expected: None, actual: None,
            backtrace: Backtrace::disabled() // NOTE: Actual capture comes later
        })
    }
    /// The span of this error, if any
    #[inline]
    pub fn span(&self) -> MaybeSpan {
        self.0.span
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
            span: MaybeSpan::Missing,
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
            MaybeSpan::Detailed(span) => {
                debug.field("span", &format_args!("{}", span));
            },
            MaybeSpan::Regular(span) => {
                debug.field("span", &format_args!("{}", span));
            }
            MaybeSpan::Missing => {}
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
            ParseErrorKind::InvalidToken => {
                f.write_str("Invalid token")?;
            }
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
            ParseErrorKind::InvalidString(ref cause) => {
                write!(f, "Invalid string ({})", cause)?;
            },
        }
        match (&self.0.expected, &self.0.actual) {
            (Some(ref expected), Some(ref actual)) => {
                write!(f, ": Expected {:?}, but got {:?}", expected, actual)?;
            },
            (Some(ref expected), None) => {
                write!(f, ": Expected {:?}", expected)?;
            },
            (None, Some(ref actual)) => {
                write!(f, ": Got {:?}", actual)?;
            },
            (None, None) => {}
        }
        match self.0.span {
            MaybeSpan::Regular(span) => {
                write!(f, " @ {}", span)?;
            },
            MaybeSpan::Detailed(span) => {
                write!(f, " @ {}", span)?;
            },
            MaybeSpan::Missing => {}
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.0.kind {
            ParseErrorKind::InvalidString(ref cause) => Some(cause),
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
