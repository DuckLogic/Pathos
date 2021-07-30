//! Expression precedence

use std::ops::{Sub, Add};

/// The precedence of an expression
///
/// From the [official docs](https://docs.python.org/3.10/reference/expressions.html#operator-precedence)
///
/// Operators in the same box group left to right (except for
/// exponentiation, which groups from right to left).
///
/// NOTE: The declaration order here is reversed
/// from the official list. We go from
/// lowest precedence (least binding power)
/// to highest precedence (most binding power).
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[repr(u8)]
pub enum ExprPrec {
    /// Specify that [PythonParser::parse_prec] has no limit to the minimum
    /// precedence it can parse.
    ///
    /// This isn't actually a valid precedence.
    NoLimit,
    /// Assignment expressions (name := val)
    Assignment,
    /// Lambdas `lambda foo: bar`
    Lambda,
    /// `1 if cond() else 4`
    Conditional,
    /// `or`
    BooleanOr,
    /// `and`
    BooleanAnd,
    /// `not x`
    BooleanNot,
    /// Comparisons, including membership tests
    /// and identity tests
    ///
    /// `in`, `not in`, `is`, `is not`, `<`, `<=`,
    /// `>`, `>=`, `!=`, `==`
    ///
    /// NOTE: These have a special left-to-right chaining feature.
    Comparisons,
    /// `|`
    BitwiseOr,
    /// `^`
    BitwiseXor,
    /// `&`
    BitwiseAnd,
    /// `<<`, `>>`
    Shifts,
    /// Addition and subtraction: `+`, `-`
    Term,
    /// Multiplication, matrix multiplication, division,
    /// floor division, remainder: `*`, `@`, `/`, `//`, `%`
    ///
    /// The `%` operator is also used for string formatting;
    /// the same precedence applies.
    Factor,
    /// Positive, negative, bitwise NOT,
    /// `+x`, `-x`, `~x`
    Unary,
    /// Exponentiation: `**`
    ///
    /// This binds less tightly than an arithmetic
    /// or bitwise unary operator on its right,
    /// that is, 2**-1 is 0.5.
    Exponentiation,
    /// An await expression: `await`
    Await,
    /// Subscription, slicing, call, attribute reference:
    /// `x[idx]`, `x[a:b]`, `x(args...)`, `x.attr`
    Call,
    /// An atomic expression, with the highest precedence.
    ///
    /// Includes binding or parenthesized expressions,
    /// list display, dictionary display, and set display.
    /// `(exprs...)`, `[exprs...]`, `{key: val...}`,
    /// `{exprs...}`
    Atom,
}
impl ExprPrec {
    /// The dummy expression precedence, used to parse target lists
    ///
    /// This is because target lists are currently implemented as expressions.
    pub const TARGET_PREC: ExprPrec = ExprPrec::BitwiseOr;
    pub const fn from_int(val: u8) -> Option<Self> {
        if (val as usize) < std::mem::variant_count::<ExprPrec>() {
            Some(unsafe { std::mem::transmute::<u8, ExprPrec>(val) })
        } else {
            None
        }
    }
}
impl Sub<u8> for ExprPrec {
    type Output = ExprPrec;

    #[inline]
    fn sub(self, rhs: u8) -> Self::Output {
        (self as u8).checked_sub(rhs).and_then(ExprPrec::from_int)
            .unwrap_or_else(|| panic!("Cannot subtract {:?} - {}", self, rhs))
    }
}
impl Add<u8> for ExprPrec {
    type Output = ExprPrec;

    #[inline]
    fn add(self, rhs: u8) -> Self::Output {
        (self as u8).checked_add(rhs).and_then(ExprPrec::from_int)
            .unwrap_or_else(|| panic!("Cannot add {:?} + {}", self, rhs))
    }
}
