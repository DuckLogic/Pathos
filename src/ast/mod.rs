use std::ops::Deref;
use std::fmt::{self, Formatter, Debug, Display};
use std::hash::{Hash, Hasher};

#[macro_use]
mod macros;
pub mod constants;
pub mod tree;

pub use self::constants::{Constant};
use crate::alloc::AllocError;
pub use crate::alloc::Allocator;

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
pub type RawIdent<'a> = crate::lexer::Ident<'a>;
#[derive(Copy, Clone)]
pub struct Ident<'a> {
    span: Span,
    raw: &'a RawIdent<'a>
}
impl<'a> Ident<'a> {
    #[inline]
    pub fn from_raw(
        span: Span,
        raw: &'a RawIdent<'a>
    ) -> Result<Self, AllocError> {
        Ok(Ident { span, raw })
    }
    #[inline]
    pub fn text(&self) -> &'a str {
        self.raw.text()
    }
    #[inline]
    pub fn as_raw(&self) -> &'a RawIdent<'a> {
        self.raw
    }
}
impl<'a> Deref for Ident<'a> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.raw.text()
    }
}
impl Debug for Ident<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let text = self.text();
        if text.bytes().all(|b| {
            b.is_ascii_alphanumeric() || b == b'_'
        }) {
            f.write_str(text)
        } else {
            write!(f, "Ident({:?})", text)
        }
    }
}
impl Display for Ident<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(self.text())
    }
}
impl PartialEq for Ident<'_> {
    #[inline]
    fn eq(&self, other: &Ident) -> bool {
        self.as_raw() == other.as_raw()
    }
}
impl Hash for Ident<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_raw().hash(state)
    }
}
impl Spanned for Ident<'_> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

/// Access the [Span] of an AST item
pub trait Spanned {
    fn span(&self) -> Span;
}



