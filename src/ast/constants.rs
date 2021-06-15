use std::fmt::{self, Write, Display, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::convert::TryFrom;

use hashbrown::HashMap;

use super::{Span, Spanned};
use crate::alloc::{Allocator, AllocError};

pub struct ConstantPool<'a> {
    arena: &'a Allocator,
    map: HashMap<&'a ConstantKind<'a>, ()>,
    none: Option<&'a ConstantKind<'a>>,
    bools: [Option<&'a ConstantKind<'a>>; 2],
}
impl<'a> ConstantPool<'a> {
    pub fn new(arena: &'a Allocator) -> Self {
        ConstantPool {
            arena, map: HashMap::new(),
            none: None, bools: [None; 2],
        }
    }
    #[inline]
    pub fn create_with(
        &mut self,
        key: ConstantKind<'a>,
        alloc: impl FnOnce(&'a Allocator) -> Result<&'a ConstantKind<'a>, AllocError>
    ) -> Result<&'a ConstantKind<'a>, AllocError> {
        use hashbrown::hash_map::{RawEntryMut};
        Ok(match self.map.raw_entry_mut()
            .from_key(&key) {
            RawEntryMut::Occupied(entry) => *entry.into_key(),
            RawEntryMut::Vacant(entry) => {
                entry.insert(alloc(self.arena)?, ()).0
            }
        })
    }
    #[inline]
    pub fn create(&mut self, key: ConstantKind<'a>) -> Result<&'a ConstantKind<'a>, AllocError> {
        self.create_with(
            key.clone(),
            move |a| Ok(&*a.alloc(key)?)
        )
    }
    #[inline]
    pub fn bool(&mut self, span: Span, b: bool) -> Result<Constant<'a>, AllocError> {
        let kind = match self.bools[b as usize] {
            Some(cached) => cached,
            None => {
                let res = self.create(ConstantKind::Bool(b))?;
                self.bools[b as usize] = Some(res);
                res
            }
        };
        Ok(Constant { span, kind })
    }
    #[inline]
    pub fn none(&mut self, span: Span) -> Result<Constant<'a>, AllocError> {
        Ok(Constant {
            span, kind: match self.none {
                Some(cached) => cached,
                None => {
                    let res = self.create(ConstantKind::None)?;
                    self.none = Some(res);
                    res
                }
            }
        })
    }
    #[inline]
    pub fn int(&mut self, span: Span, val: i64) -> Result<Constant<'a>, AllocError> {
        Ok(Constant {
            span, kind: self.create(ConstantKind::Integer(val))?
        })
    }
    #[inline]
    pub fn big_int(&mut self, span: Span, val: &'a BigInt) -> Result<Constant<'a>, AllocError> {
        Ok(Constant {
            span, kind: self.create(ConstantKind::BigInteger(val))?
        })
    }
    #[inline]
    pub fn float(&mut self, span: Span, f: FloatLiteral) -> Result<Constant<'a>, AllocError> {
        Ok(Constant {
            span, kind: self.create(ConstantKind::Float(f))?
        })
    }
    #[inline]
    pub fn string(&mut self, span: Span, lit: StringLiteral<'a>) -> Result<Constant<'a>, AllocError> {
        Ok(Constant {
            span, kind: self.create(ConstantKind::String(lit))?
        })
    }

    #[inline]
    pub fn complex(&mut self, span: Span, lit: ComplexLiteral) -> Result<Constant<'a>, AllocError> {
        Ok(Constant {
            span, kind: self.create(ConstantKind::Complex(lit))?
        })
    }
}
impl Debug for ConstantPool<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        /*
         * I'm not even going to list the interned constants
         * I feel that it would be a waste of space.
         */
        f.debug_struct("ConstantPool").finish()
    }
}
#[derive(Copy, Clone)]
pub struct Constant<'a> {
    span: Span,
    kind: &'a ConstantKind<'a>
}
impl<'a> Constant<'a> {
    #[inline]
    pub fn kind(self) -> &'a ConstantKind<'a> {
        self.kind
    }
}
impl<'a> Eq for Constant<'a> {}
impl<'a> PartialEq for Constant<'a> {
    #[inline]
    fn eq(&self, other: &Constant) -> bool {
        debug_assert_eq!(
            std::ptr::eq(self.kind, other.kind),
            self.kind == other.kind,
            concat!(
                "Pointer equality and value equality ",
                " gave different results for {:?} and {:?}"),
            self.kind, other.kind
        );
        std::ptr::eq(self.kind, other.kind)
    }
}
impl Hash for Constant<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Because we expect that values are interned,
        // we can use the pointer hash
        std::ptr::hash(self.kind, state)
    }
}
impl Display for Constant<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self.kind(), f)
    }
}
impl Debug for Constant<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?} @ {}", self.kind, self.span)
    }
}
impl<'a> Deref for Constant<'a> {
    type Target = ConstantKind<'a>;
    #[inline]
    fn deref(&self) -> &ConstantKind<'a> {
        &self.kind
    }
}
impl Spanned for Constant<'_> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstantKind<'a> {
    None,
    Bool(bool),
    Tuple(&'a [Constant<'a>]),
    Integer(i64),
    BigInteger(&'a BigInt),
    Float(FloatLiteral),
    String(StringLiteral<'a>),
    Complex(ComplexLiteral)
}
impl Display for ConstantKind<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            ConstantKind::None => {
                f.write_str("None")
            },
            ConstantKind::Bool(true) => f.write_str("True"),
            ConstantKind::Bool(false) => f.write_str("False"),
            ConstantKind::Tuple(ref vals) => {
                f.write_str("(")?;
                for val in *vals {
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
#[derive(Debug, Copy, Clone)]
pub struct StringLiteral<'a> {
    /// The value of the string, with all escapes,
    /// newlines and backslashes fully interpreted
    pub value: &'a str,
    pub style: StringStyle,
}
impl<'a> StringLiteral<'a> {
    #[inline]
    pub fn prefix(&self) -> Option<StringPrefix> {
        self.style.prefix
    }
}
impl<'a> Eq for StringLiteral<'a> {}
impl<'a> PartialEq for StringLiteral<'a> {
    fn eq(&self, other: &StringLiteral) -> bool {
        self.prefix() == other.prefix() && self.value == other.value
    }
}
impl<'a> Hash for StringLiteral<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.prefix().hash(h);
        h.write(self.value.as_bytes());
    }
}
impl<'a> Display for StringLiteral<'a> {
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
    #[inline]
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
    pub fn len(self) -> usize {
        if self.is_triple_string() { 3 } else { 1 }
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
