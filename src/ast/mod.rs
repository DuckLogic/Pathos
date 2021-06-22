use std::fmt::{self, Formatter, Debug, Display};
#[cfg(feature = "serialize")]
use serde_with::{DeserializeFromStr, SerializeDisplay};

use thiserror::Error;

pub mod constants;
pub mod tree;
pub mod ident;

pub use self::constants::{Constant};
pub use crate::alloc::Allocator;
pub use self::ident::{Ident, Symbol};
use crate::ast::tree::{ExprKind, Expr};
use crate::alloc::AllocError;
use std::str::FromStr;
use std::hash::Hash;

#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(SerializeDisplay, DeserializeFromStr))]
pub struct Span {
    pub start: usize,
    pub end: usize
}
impl Span {
    /// Create a dummy span for debugging purposes
    ///
    /// **WARNING**: The resulting span is not distinguishable
    /// from normally created spans
    #[inline]
    pub const fn dummy() -> Span {
        Span { start: 0, end: 0 }
    }
    /// Expand to include the other span
    #[inline]
    pub fn expand(&self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end)
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
        write!(f, "{}..{}", self.start, self.end)
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
        end: usize,
        start: usize
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
        let start = usize::from_str(&s[..dot_start_index])?;
        let end = usize::from_str(&s[dot_end_index..])?;
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

/// A [ParseVisitor](crate::parse::visitor::ParseVisitor) that constructs
/// an abstract syntax tree
#[derive(Copy, Clone, Debug)]
pub struct AstBuilder<'a> {
    pub arena: &'a Allocator,
}
impl<'a> AstBuilder<'a> {
    /// Create an expression of the specified [ExprKind]
    #[inline]
    pub fn expr(&self, kind: ExprKind<'a>) -> Result<Expr<'a>, AllocError> {
        Ok(self.arena.alloc(kind)?)
    }
}
