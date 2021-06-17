use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt::{Display, Formatter, Debug};
use std::fmt;

use crate::alloc::AllocError;
use crate::ast::Span;
use crate::lexer::StringError;

#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    InvalidToken,
    AllocationFailed,
    UnexpectedEof,
    UnexpectedToken,
    InvalidString(StringError),
}

#[derive(Debug)]
struct ParseErrorInner {
    /// The span of the source location
    ///
    /// There are only two instances where this can be `None`:
    /// 1. Out of memory errors
    /// 2. A ParseVisitError
    span: Option<Span>,
    expected: Option<String>,
    actual: Option<String>,
    kind: ParseErrorKind,
    backtrace: Backtrace
}

pub struct ParseError(Box<ParseErrorInner>);
impl ParseError {
    /// Give additional context on the type of item that was "expected"
    #[cold]
    pub fn with_expected_msg<T: ToString>(mut self, msg: T) -> Self {
        self.0.expected = Some(msg.to_string());
        self
    }
    #[inline]
    pub fn builder(span: Span, kind: ParseErrorKind) -> ParseErrorBuilder {
        ParseErrorBuilder(ParseErrorInner {
            span: Some(span), kind,
            expected: None, actual: None,
            backtrace: Backtrace::disabled() // NOTE: Actual capture comes later
        })
    }
    /// The span of this error, if any
    #[inline]
    pub fn span(&self) -> Option<Span> {
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
            span: None,
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
        if let Some(span) = self.0.span {
            debug.field("span", &format_args!("{}", span));
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
        if let Some(span) = self.0.span {
            write!(f, " @ {}", span)?;
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
