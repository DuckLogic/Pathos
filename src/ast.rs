use std::convert::TryFrom;
use std::ops::Deref;
use std::fmt::{self, Formatter, Write, Debug, Display};
use std::sync::Arc;
use std::hash::{Hash, Hasher};

use educe::Educe;

use crate::BigInt;
use crate::lexer::StringPrefix;
use crate::lexer::{StringStyle, QuoteStyle};


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
#[derive(Clone)]
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
impl PartialEq for Constant {
    fn eq(&self, other: &Constant) -> bool {
        Arc::ptr_eq(&self.kind, &other.kind) 
            || self.kind == other.kind
    }
}
impl Hash for Constant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}
impl Display for Constant {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self.kind(), f)
    }
}
impl Debug for Constant {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?} @ {}", self.kind, self.span)
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

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum ConstantKind {
    Tuple(Vec<Constant>),
    Integer(i64),
    BigInteger(BigInt),
    Float(FloatLiteral),
    String(StringLiteral),
    Complex(ComplexLiteral)
}
impl Display for ConstantKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            ConstantKind::Tuple(ref vals) => {
                f.write_str("(")?;
                for val in vals {
                    write!(f, "{},", val)?;
                }
                f.write_str(")")?;
                Ok(())
            },
            ConstantKind::Integer(val) => {
                write!(f, "{}", val)
            },
            ConstantKind::BigInteger(ref val) => {
                write!(f, "{}", val)
            },
            ConstantKind::Float(flt) => {
                write!(f, "{}", flt)
            },
            ConstantKind::String(ref s) => {
                write!(f, "{}", s)
            },
            ConstantKind::Complex(cplx) => {
                write!(f, "{}", cplx)
            }
        }
    }
}

/// A marker type to indicate a float literal is invalid
///
/// In python, this can occur if a float is infinite or NaN.
pub struct InvalidFloatLiteral(());

/// A python floating-point literal
///
/// These are notable for the fact that NaN and infinity
/// are forbidden.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Default)]
pub struct FloatLiteral(::ordered_float::NotNan<f64>);
impl FloatLiteral {
    /// Create a new floating point literal from 
    /// the specified value.
    ///
    /// Returns an error if the float is infinity or NaN
    #[inline]
    pub fn new(value: f64) -> Result<FloatLiteral, InvalidFloatLiteral> {
        if value.is_finite() {
            Ok(FloatLiteral(::ordered_float::NotNan::new(value).unwrap()))
        } else {
            Err(InvalidFloatLiteral(()))
        }
    }
    /// The value of the literal
    #[inline]
    pub fn value(self) -> f64 {
        self.0.into_inner()
    }
    /// Convert into an [not-NAN ordered float][::ordered_float::NotNan]
    #[inline]
    pub const fn into_ordered_float(self) -> ::ordered_float::NotNan<f64> {
        self.0
    }
}
impl Deref for FloatLiteral {
    type Target = f64;
    #[inline]
    fn deref(&self) -> &f64 {
        &*self.0
    }
}
impl From<FloatLiteral> for f64 {
    #[inline]
    fn from(lit: FloatLiteral) -> f64 {
        lit.value()
    }
}
impl From<FloatLiteral> for ::ordered_float::NotNan<f64> {
    #[inline]
    fn from(lit: FloatLiteral) -> ::ordered_float::NotNan<f64> {
        lit.0
    }
}
impl TryFrom<f64> for FloatLiteral {
    type Error = InvalidFloatLiteral;
    #[inline]
    fn try_from(val: f64) -> Result<FloatLiteral, Self::Error> {
        FloatLiteral::new(val)
    }
}
impl Display for FloatLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}
impl Debug for FloatLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self.value())
    }
}

#[derive(Copy, PartialEq, Eq, Hash, Clone, Debug)]
pub struct ComplexLiteral {
    pub real: Option<FloatLiteral>,
    pub imaginary: FloatLiteral
}
impl Display for ComplexLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if let Some(real) = self.real {
            write!(f, "({} + ", real)?;
        }
        write!(f, "{}j", self.imaginary)?;
        if self.real.is_some() {
            f.write_str(")")?;
        }
        Ok(())
    }
}

/// A string literal, including style information.
///
/// NOTE: Equality is based only on value (and prefix).
/// It ignores stylistic information like the difference
/// between single and triple quotes.
#[derive(Debug, Clone)]
pub struct StringLiteral {
    pub value: String,
    pub style: StringStyle,
}
impl StringLiteral {
    #[inline]
    pub fn prefix(&self) -> Option<StringPrefix> {
        self.style.prefix
    }
}
impl Eq for StringLiteral {}
impl PartialEq for StringLiteral {
    fn eq(&self, other: &StringLiteral) -> bool {
        self.prefix() == other.prefix() && self.value == other.value
    }
}
impl Hash for StringLiteral {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.prefix().hash(h);
        h.write(self.value.as_bytes());
    }
}
impl Display for StringLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        /*
         * NOTE: Ignore raw strings.
         * They can't necessarily represent all possible escapes
         */
        if let Some(prefix) = self.style.prefix {
            f.write_char(prefix.prefix_char())?;
        }
        f.write_str(self.style.quote_style.text())?;
        for c in self.value.chars() {
            self.style.quote_style.escape_char(
                c, f,
                crate::lexer::UnicodeEscapeStyle::EXPLICIT
            );
        }
        f.write_str(self.style.quote_style.text())?;
        Ok(())
    }
}


pub trait Spanned {
    fn span(&self) -> Span;
}

include!(concat!(env!("OUT_DIR"), "/ast_gen.rs"));
