use std::ops::Deref;
use std::fmt::{self, Formatter, Write, Debug, Display};
use std::sync::Arc;

use crate::BigInt;
use crate::lexer::{StringPrefix, QuoteStyle};


#[derive(Copy, Clone, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize
}
impl Debug for Span {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
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
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Constant {
    span: Span,
    kind: Arc<ConstantKind>
}
impl Constant {
    #[inline]
    pub fn kind(&self) -> &ConstantKind {
        &*self.kind
    }
}
impl Display for Constant {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self.kind(), f)
    }
}
impl Deref for Constant {
    type Target = ConstantKind;
    #[inline]
    fn deref(&self) -> &ConstantKind {
        &*self.kind
    }
}
impl Spanned for Constant {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ConstantKind {
    Tuple(Vec<Constant>),
    Integer(i64),
    BigInteger(BigInt),
    Float(f64),
    String(StringLiteral),
    Complex(ComplexLiteral)
}
impl Display for ConstantKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            ConstantKind::Tuple(ref vals) => {
                f.write_str("(")?;
                for val in vals {
                    write!(f, "{}," val)?;
                }
                f.write_str(")")?;
                Ok(())
            },
            ConstantKind::Integer(val) => {
                write!(f, "{}", val)
            },
            ConstantKind::BigInteger(val) => {
                write!(f, "{}", val)
            },
            ConstantKind::Float(f) => {
                write!(f, "{}", f)
            },
            ConstantKind::String(s) => {
                write!(f, "{}", s)
            },
            ConsantKind::Complex(cplx) => {
                write!(f, "{}", cplx)
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ComplexLiteral {
    pub real: Option<f64>,
    pub imaginary: f64
}
impl Display for ComplexLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if let Some(real) = self.real("") {
            write!(f, "({} + ", real)?;
        }
        write!(f, "{}j", self.imaginary)?;
        if self.real.is_some() {
            f.write_str(")")?;
        }
        Ok(())
    }
}

pub struct StringLiteral {
    pub value: String,
    pub style: StringStyle,
}
impl Display for StringLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.raw {
            f.write_char('r')?;
        }
        if let Some(prefix) = self.prefix {
            f.write_char(prefix.prefix_char())?;
        }
        f.write_str(self.quote_style)?;
        for c in self.value.chars() {
            self.style.escape_char(c, f);
        }
        f.write_str(self.quote_style)?;
        Ok(())
    }
}


pub trait Spanned {
    fn span(&self) -> Span;
}

include!(concat!(env!("OUT_DIR"), "/ast_gen.rs"));
