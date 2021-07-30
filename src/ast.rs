use std::ops::Range;
use std::fmt::{self, Debug, Display, Formatter};
use std::convert::{TryInto, TryFrom};
use std::hash::Hash;
use std::str::FromStr;

#[cfg(feature = "serialize")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "serialize")]
use serde_with::{SerializeDisplay, DeserializeFromStr};
use thiserror::Error;

pub use crate::ast::ident::{Ident, Symbol};

pub mod ident;

/// Represents a single position in the source code.
///
/// Currently implemented as a byte-index
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Position(pub u64);

impl Position {
    #[track_caller]
    pub fn index(&self) -> usize {
        match usize::try_from(self.0) {
            Ok(val) => val,
            Err(_) => panic!("Position overflowed usize: {}", self.0)
        }
    }
}

impl Position {
    /// Create a dummy position for use with debugging
    #[inline]
    pub const fn dummy() -> Position {
        Position(u64::MAX)
    }
    #[inline]
    pub fn is_dummy(&self) -> bool {
        self.0 == u64::MAX
    }
    #[inline]
    pub fn with_len<L: TryInto<u64>>(self, len: L) -> Span
        where L::Error: Debug {
        let len = len.try_into().unwrap();
        Span {
            start: self,
            end: Position(self.0 + len)
        }
    }
}
impl From<u64> for Position {
    #[inline]
    fn from(i: u64) -> Self {
        Position(i)
    }
}
impl From<usize> for Position {
    #[inline]
    fn from(i: usize) -> Self {
        Position(i as u64)
    }
}
impl Display for Position {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_dummy() {
            f.write_str("DUMMY")
        } else {
            Display::fmt(&self.0, f)
        }
    }
}
impl Debug for Position {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(SerializeDisplay, DeserializeFromStr))]
pub struct Span {
    pub start: Position,
    pub end: Position
}
impl Span {
    #[inline]
    pub fn from_logos_span(span: Range<usize>) -> Span {
        Span {
            start: span.start.into(),
            end: span.end.into()
        }
    }
    /// Create a dummy span for debugging purposes
    ///
    /// **WARNING**: The resulting span is not distinguishable
    /// from normally created spans
    #[inline]
    pub const fn dummy() -> Span {
        Span { start: Position::dummy(), end: Position::dummy() }
    }
    #[inline]
    pub fn is_dummy(&self) -> bool {
        self.start.is_dummy() && self.end.is_dummy()
    }
    /// Expand to include the other span
    #[inline]
    pub fn expand(&self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end)
        }
    }
    /// Offset this span by the specified amount
    #[inline]
    pub fn offset(&self, amount: u64) -> Span {
        Span {
            start: Position(self.start.0 + amount),
            end: Position(self.end.0 + amount)
        }
    }
}
impl Debug for Span {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}
impl Display for Span {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.is_dummy() {
            f.write_str("DUMMY")
        } else {
            write!(f, "{}..{}", self.start, self.end)
        }
    }
}
#[derive(Error, Debug)]
pub enum SpanParseError {
    #[error("Invalid integer: {0}")]
    InvalidInt(#[from] std::num::ParseIntError),
    #[error("Missing dots '..'")]
    MissingDots,
    #[error("Expected 2 or 3 dots, but got {actual}")]
    TooManyDots {
        actual: usize
    },
    #[error("The end {end} must be <= start {start} ")]
    EndBeforeStart {
        end: Position,
        start: Position
    }
}
impl FromStr for Span {
    type Err = SpanParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (dot_start_index, dot_end_index) = match s.find("..") {
            Some(start_index) => {
                let remaining = &s[start_index..];
                let num_dots = remaining.find(|c: char| c != '.').unwrap_or(remaining.len());
                assert!(num_dots >= 2);
                if num_dots > 3 {
                    return Err(SpanParseError::TooManyDots {
                        actual: num_dots
                    })
                } else {
                    (start_index, start_index + num_dots)
                }
            },
            None => return Err(SpanParseError::MissingDots)
        };
        let start = u64::from_str(&s[..dot_start_index])?.into();
        let end = u64::from_str(&s[dot_end_index..])?.into();
        if start <= end {
            Ok(Span { start, end })
        } else {
            Err(SpanParseError::EndBeforeStart {
                end, start
            })
        }
    }
}

/// Dummy trait that indicates a type supports serde's [Serialize](::serde::Serialize)
///
/// NOTE: Support for `Deserialize` is ignored
///
/// This is needed so we can hide serde support behind a feature flag.
/// It is automatically implemented when a type supports both serialization and deserialization.
#[cfg(feature = "serialize")]
pub trait Serializable: ::serde::Serialize {}

#[cfg(feature = "serialize")]
impl<T: ::serde::Serialize> Serializable for T {}

/// Dummy trait that is used when serialization is enabled
///
/// This is needed because we hide serde support behind a feature flag.
/// When support is disabled (as it is now), this is automatically implemented for all types.
#[cfg(not(feature = "serialize"))]
pub trait Serializable {}

#[cfg(not(feature = "serialize"))]
impl<T> Serializable for T {}

/// Indicates a node in the abstract syntax tree
pub trait AstNode: Spanned + Eq + Hash + Debug + Clone + Serializable {
}
/// Access the [Span] of an AST item
pub trait Spanned {
    fn span(&self) -> Span;
}
