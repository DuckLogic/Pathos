use std::sync::Arc;
use std::fmt::{self, Write, Display, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::convert::TryFrom;

use super::{Span, Spanned};

#[derive(Clone)]
pub struct Constant {
    span: Span,
    kind: Arc<ConstantKind>
}
impl Constant {
    #[inline]
    pub fn new(span: Span, kind: ConstantKind) -> Constant {
        Constant { span, kind: Arc::new(kind) }
    }
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

pub trait ConstantVisitor<'a> {
    type Constant;
    fn visit_none(&mut self, span: Span) -> Self::Constant;
    fn visit_bool(&mut self, span: Span, val: bool) -> Self::Constant;
    fn visit_tuple(
        &mut self,
        span: Span,
        iter: impl Iterator<Item=Self::Constant>
    ) -> Self::Constant;
    fn visit_int(&mut self, span: Span, val: i64) -> Self::Constant;
    fn visit_big_int(&mut self, span: Span, val: &'a BigInt) -> Self::Constant;
    fn visit_float(&mut self, span: Span, val: f64) -> Self::Constant;
    fn visit_string(&mut self, span: Span, lit: crate::lexer::StringInfo<'a>) -> Self::Constant;
    fn visit_complex(
        &mut self,
        span: Span,
        real: Option<(Span, f64)>,
        imaginary: (Span, f64),
    ) -> Self::Constant;
}
impl<'a> ConstantVisitor<'a> for super::mem_visitor::MemoryVisitor {
    type Constant = Constant;
    fn visit_none(&mut self, span: Span) -> Self::Constant {
        Constant::new(span, ConstantKind::None)
    }
    fn visit_bool(&mut self, span: Span, val: bool) -> Self::Constant {
        Constant::new(span, ConstantKind::Bool(val))
    }
    fn visit_tuple(
        &mut self,
        span: Span,
        iter: impl Iterator<Item=Self::Constant>
    ) -> Self::Constant {
        Constant::new(span, ConstantKind::Tuple(iter.collect()))
    }
    fn visit_int(&mut self, span: Span, val: i64) -> Self::Constant {
        Constant::new(span, ConstantKind::Integer(val))
    }
    fn visit_big_int(&mut self, span: Span, val: &'a BigInt) -> Self::Constant {
        Constant::new(span, ConstantKind::BigInteger(val.clone()))
    }
    fn visit_float(&mut self, span: Span, val: f64) -> Self::Constant {
        Constant::new(span, ConstantKind::Float(FloatLiteral::new(val).unwrap()))
    }
    fn visit_string(
        &mut self,
        span: Span,
        lit: crate::lexer::StringInfo<'a>
    ) -> Self::Constant {
        Constant::new(span, ConstantKind::String(StringLiteral {
            value: lit.original_text.into(),
            style: lit.style
        }))
    }
    fn visit_complex(
        &mut self,
        span: Span,
        real: Option<(Span, f64)>,
        imaginary: (Span, f64),
    ) -> Self::Constant {
        Constant::new(span, ConstantKind::Complex(ComplexLiteral {
            real: real.map(|(_, val)| FloatLiteral::new(val).unwrap()),
            imaginary: FloatLiteral::new(imaginary.1).unwrap()
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum ConstantKind {
    None,
    Bool(bool),
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
#[derive(Debug, Clone, Copy)]
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
                UnicodeEscapeStyle::EXPLICIT
            )?;
        }
        f.write_str(self.style.quote_style.text())?;
        Ok(())
    }
}

/// Style information for a string
///
/// This also includes its prefix
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct StringStyle {
    /// An explicit string prefix,
    /// or none if it's just a plain string
    pub prefix: Option<StringPrefix>,
    /// True if the string is also a raw string
    pub raw: bool,
    pub quote_style: QuoteStyle
}

/// The prefix for a string
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum StringPrefix {
    /// A format string prefix.
    Formatted,
    /// An explicit unicode prefix.
    ///
    /// Note that this is redundant on Python 3. 
    Unicode,
    /// A byte string prefix
    Bytes
}
impl StringPrefix {
    pub fn prefix_char(self) -> char {
        match self {
            StringPrefix::Formatted => 'f',
            StringPrefix::Unicode => 'u',
            StringPrefix::Bytes => 'b'
        }
    }
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QuoteStyle {
    Double,
    Single,
    DoubleLong,
    SingleLong,
}
#[non_exhaustive]
pub enum UnicodeEscapeStyle {
    Simple,
    ForceEscape
}
impl UnicodeEscapeStyle {
    pub const EXPLICIT: UnicodeEscapeStyle = UnicodeEscapeStyle::ForceEscape;
}
impl Default for UnicodeEscapeStyle {
    #[inline]
    fn default() -> UnicodeEscapeStyle {
        UnicodeEscapeStyle::Simple
    }
}
impl QuoteStyle {
    pub fn escape_char<F: std::fmt::Write>(
        &self, c: char,
        out: &mut F,
        unicode_style: UnicodeEscapeStyle
    ) -> fmt::Result {
        match c {
            '\t' => out.write_str("\\t"),
            '\r' => out.write_str("\\r"),
            '\n' => out.write_str("\\n"),
            '"' => {
                if self.start_char() == '"' {
                    out.write_str("\\\"")
                } else {
                    out.write_char('"')
                }
            },
            '\'' => {
                if self.start_char() == '\'' {
                    out.write_str("\\'")
                } else {
                    out.write_char('\'')
                }
            },
            '\\' => {
                out.write_str("\\\\")
            },
            '\u{20}'..='\u{7e}' => {
                out.write_char(c)
            },
            _ => {
                match unicode_style {
                    UnicodeEscapeStyle::Simple => {
                        if c.is_alphanumeric() {
                            return out.write_char(c);
                        }
                    },
                    UnicodeEscapeStyle::ForceEscape => {}
                }
                // Fallthrough to unicode escape
                if c as u64 <= 0xFFFF {
                    write!(out, "\\u{:04X}", c as u64)
                } else {
                    assert!(c as u64 <= 0xFFFF_FFFF);
                    write!(out, "\\U{:08X}", c as u64)
                }
            }
        }
    }
    #[inline]
    pub fn start_byte(self) -> u8 {
        match self {
            QuoteStyle::Single |
            QuoteStyle::SingleLong => b'\'',
            QuoteStyle::Double |
            QuoteStyle::DoubleLong => b'"',
        }
    }
    #[inline]
    pub fn start_char(self) -> char {
        self.start_byte() as char
    }
    #[inline]
    pub fn text(self) -> &'static str {
        match self {
            QuoteStyle::DoubleLong => r#"""""#,
            QuoteStyle::SingleLong => r"'''",
            QuoteStyle::Double => r#"""#,
            QuoteStyle::Single => r"'",
        }
    }
    #[inline]
    pub fn is_triple_string(self) -> bool {
        match self {
            QuoteStyle::Single |
            QuoteStyle::Double => false,
            QuoteStyle::SingleLong |
            QuoteStyle::DoubleLong => true,
        }
    }
}

/// An arbitrary precision integer
#[cfg(all(feature="num-bigint", not(feature="rug")))]
pub type BigInt = num_bigint::BigInt;
/// A rug BigInt
#[cfg(feature="rug")]
pub type BigInt = rug::Integer;
/// Fallback arbitrary precision integers,
/// when all dependencies are disabled
///
/// Stored as plain text
#[cfg(not(any(feature="num-bigint", feature="rug")))]
pub struct BigInt {
    /// The raw plain text
    pub text: String
}
