use std::fmt::{self, Formatter, Debug, Display};

#[macro_use]
mod macros;
pub mod constants;
pub mod tree;
pub mod ident;

pub use self::constants::{Constant};
pub use crate::alloc::Allocator;
pub use self::ident::Ident;

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



