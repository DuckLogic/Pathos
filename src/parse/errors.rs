use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fmt;

use crate::alloc::AllocError;
use crate::ast::Span;
use crate::lexer::StringError;

#[derive(Debug, Clone)]
pub enum ParseErrorKind<VE: ParseVisitError = !> {
    InvalidToken,
    AllocationFailed,
    UnexpectedEof,
    UnexpectedToken,
    InvalidString(StringError),
    /// An error that occurred while visiting
    VisitError(VE::ExtraVariant)
}

#[derive(Debug)]
struct ParseErrorInner<VE: ParseVisitError> {
    /// The span of the source location
    ///
    /// There are only two instances where this can be `None`:
    /// 1. Out of memory errors
    /// 2. A ParseVisitError
    span: Option<Span>,
    expected: Option<String>,
    actual: Option<String>,
    kind: ParseErrorKind<VE>,
    backtrace: Backtrace
}

/// A marker trait, indicating an error caused by a [ParseVisitor](std::parse::visitor::ParseVisitor)
///
/// These are distinct from normal [ParseError]s, because they are caused by user code
/// and not an actual syntax error.
///
/// Technically, this is a conversion trait. It
pub trait ParseVisitError: std::error::Error {
    /// The extra variant added to [ParseErrorKind]
    ///
    /// Generally, this is Self, but sometimes ParseError
    /// without needing to "add an extra variant".
    ///
    /// For example `impl ParseVisitError for AllocError`
    /// has `ExtraVariant = !`, because ParseErrorKind already
    /// has a variant indicating out of memory (so there is no need for another one).
    type ExtraVariant: Error;
    /// Convert this visit error into the appropriate parse error
    fn into_parse_error(self) -> ParseError<Self>;
}
impl ParseVisitError for AllocError {
    type ExtraVariant = !;
    #[inline]
    #[track_caller]
    fn into_parse_error(self) -> ParseError<Self> {
        ParseError::from_failed_alloc(self)
    }
}

#[derive(Debug)]
pub struct ParseError<VE: ParseVisitError = !>(Box<ParseErrorInner<VE>>);

impl<VE: ParseVisitError> ParseError<VE> {
    /// Create a parse error indicating an out of memory condition
    #[cold]
    #[track_caller]
    pub fn from_failed_alloc(_cause: AllocError) -> Self {
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
    /// Give additional context on the type of item that was "expected"
    #[cold]
    pub fn with_expected_msg<T: ToString>(mut self, msg: T) -> Self {
        self.0.expected = Some(msg.to_string());
        self
    }
    #[inline]
    pub fn builder(span: Span, kind: ParseErrorKind<VE>) -> ParseErrorBuilder<VE> {
        ParseErrorBuilder(ParseErrorInner {
            span: Some(span), kind,
            expected: None, actual: None,
            backtrace: Backtrace::disabled() // NOTE: Actual capture comes later
        })
    }
}
impl<VE: ParseVisitError> From<VE> for ParseError<VE> {
    #[inline]
    #[cold]
    #[track_caller]
    fn from(cause: VE) -> Self {
        cause.into_parse_error()
    }
}

impl<VE: ParseVisitError> Display for ParseError<VE> {
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
            ParseErrorKind::VisitError(ref extra) => {
                write!(f, "{}", extra)?;
                return Ok(()); // We don't write anything else. The user has complete control
            }
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

impl<VE: ParseVisitError> std::error::Error for ParseError<VE> {
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

pub struct ParseErrorBuilder<VE: ParseVisitError = !>(ParseErrorInner<VE>);

impl<VE: ParseVisitError> ParseErrorBuilder<VE> {
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
    pub fn build(mut self) -> ParseError<VE> {
        ParseError(Box::new(ParseErrorInner {
            span: self.0.span,
            kind: self.0.kind,
            expected: self.0.expected.take(),
            actual: self.0.actual.take(),
            backtrace: Backtrace::capture()
        }))
    }
}
