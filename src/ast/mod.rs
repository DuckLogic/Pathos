use std::fmt::{self, Formatter, Debug, Display};

pub mod constants;
pub mod tree;
pub mod ident;

pub use self::constants::{Constant};
pub use crate::alloc::Allocator;
pub use self::ident::Ident;
use crate::ast::tree::{ExprKind, Expr};
use crate::alloc::AllocError;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize
}
impl Span {
    /// Create a dummy span for debugging purposes
    ///
    /// NOTE: The resulting span is not distinguishable
    /// from
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
