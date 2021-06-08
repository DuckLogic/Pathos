use std::ops::Deref;
use std::fmt::{self, Formatter, Debug, Display};
use std::sync::Arc;
use std::hash::{Hash, Hasher};

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
struct IdentInner<'a> {
    span: Span,
    // NOTE: Double boxing avoids fat pointer
    raw: RawIdent<'a>
}
#[derive(Clone)]
pub struct Ident<'a>(&'a IdentInner<'a>);
impl<'a> Ident<'a> {
    #[inline]
    pub fn from_raw(
        arena: &'a Allocator, span: Span,
        raw: RawIdent<'a>
    ) -> Result<Self, AllocError> {
        Ok(arena.alloc(IdentInner {
            span, raw
        }))
    }
    #[inline]
    pub fn text(&self) -> &'a str {
        self.0.text()
    }
    #[inline]
    pub fn as_raw(&self) -> &'a RawIdent<'a> {
        self.0
    }
}
impl<'a> Deref for Ident<'a> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.0.text()
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
        self.as_raw().span
    }
}

/// Access the [Span] of an AST item
pub trait Spanned {
    fn span(&self) -> Span;
}



