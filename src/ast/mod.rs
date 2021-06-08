use std::ops::Deref;
use std::fmt::{self, Formatter, Debug, Display};
use std::sync::Arc;
use std::hash::{Hash, Hasher};

pub mod constants;
pub mod visitor;
pub mod mem_visitor;

pub use self::visitor::AstVisitor;
pub use self::constants::{Constant};

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
struct IdentInner {
    span: Span,
    // NOTE: Double boxing avoids fat pointer
    text: Box<str>   
}
#[derive(Clone)]
pub struct Ident(Arc<IdentInner>);
impl Ident {
    pub fn new(span: Span, text: impl Into<Box<str>>) -> Self {
        Ident(Arc::new(IdentInner {
            span, text: text.into()
        }))
    }
    #[inline]
    pub fn text(&self) -> &str {
        &self.0.text
    }
}
impl Deref for Ident {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        &self.0.text
    }
}
impl Debug for Ident {
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
impl Display for Ident {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(self.text())
    }
}
impl PartialEq for Ident {
    fn eq(&self, other: &Ident) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || self.text() == other.text()
    }
}
impl Hash for Ident {
    fn hash<H: Hasher>(&self, h: &mut H) {
        h.write(self.text().as_bytes());
    }
}
impl Spanned for Ident {
    #[inline]
    fn span(&self) -> Span {
        self.0.span
    }
}

/// Access the [Span] of an AST item
pub trait Spanned {
    fn span(&self) -> Span;
}


/// A in-memory representation of the AST
///
/// This is automatically generated from the ASDL file
pub mod tree {
    use super::*;
    use super::constants::ConstantVisitor;
    use educe::Educe;
    include!(concat!(env!("OUT_DIR"), "/ast_gen.rs"));
}
