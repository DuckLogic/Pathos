use std::iter;
use std::marker::PhantomData;

use crate::lexer::{Token};
use crate::ast::AstVisitor;

use super::parser::{SpannedToken, Parser, ParseError};
use super::PythonParser;

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
enum ExprPrec {
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
    Exponentation,
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
    const MAX: ExprPrec = ExprPrec::Atom;
    const MIN: ExprPrec = ExprPrec::Assignment;
}

impl<'src, 'a, 'v, V: AstVisitor<'a>> PythonParser<'src, 'a, 'v, V> {
    pub fn expression(&mut self) -> Result<V::Expr, ParseError> {
        self.parse_prec(ExprPrec::Atom)
    }
    /// A pratt parser for python expressions
    fn parse_prec(&mut self, prec: ExprPrec) -> Result<V::Expr, ParseError>{
        let token = self.parser.peek();
        let left = match token.and_then(Self::prefix_parser) {
            Some(func) => {
                self.parser.skip()?;
                func(&mut *self, &token)
            },
            None => {
                return Err(self.parser.unexpected("an expression"))
            }
        };
        let next_token = match self.parser.peek() {
            Some(tk) => tk, 
            None => return Ok(left)
        };
        match Self::infix_parser(next_token.into()) {
            Some(func) => {
                func(self, left, &next_token)
            },
            None => Ok(left) // just give left
        }
    }
    fn prefix_parser(token: &Token<'a>) 
        -> Option<fn(&mut Self, &SpannedToken<'a>) -> Result<V::Expr, ParseError>> {
        Some(match *token {
            Token::Ident(_) => Self::name,
            Token::True | Token::False |
            Token::None | Token::StringLiteral(_) |
            Token::IntegerLiteral(_) |
            Token::BigIntegerLiteral(_) |
            Token::FloatLiteral(_) => Self::constant,
            /*
             * For constructing a list, a set or a dictionary
             * Python provides special syntax called “displays”,
             * each of them in two flavors:
             * - either the container contents are listed explicitly, or
             * - they are computed via a set of looping and filtering
             *   instructions, called a comprehension.
             */
            Token::OpenBracket => Self::list,
            _ => return None
        })
    }
    fn infix_parser(token: &Token<'a>) 
        -> Option<fn(&mut Self, V::Expr, &SpannedToken<'a>) -> Result<V::Expr, ParseError>> {
            None
    }
    fn name(&mut self, tk: &SpannedToken<'a>) -> Result<V::Expr, ParseError> {
        let ident = match **tk {
            Token::Ident(inner) => {
                self.visitor.visit_ident(
                    tk.span,
                    *inner
                );
            }, 
            _ => unreachable!()
        };
        Ok(self.visitor.visit_expr_name(
            ident.span,
            ident,
            self.expression_context
        ))
    }
    fn constant(&mut self, tk: &SpannedToken<'a>) -> Result<V::Expr, ParseError> {
        let span = tk.span;
        let kind = None;
        let constant = match tk.kind {
            Token::True => self.visitor.visit_bool(span, true),
            Token::False => self.visitor.visit_bool(span, false),
            Token::None => self.visitor.visit_none(span),
            Token::StringLiteral(lit) => {
                match lit.style.prefix {
                    Some(crate::ast::constants::StringPrefix::Unicode) => {
                        kind = Some("u");
                    },
                    _ => {}
                }
                self.visitor.visit_string(span, lit)
            },
            Token::IntegerLiteral(val) => {
                self.visitor.visit_int(span, val)
            },
            Token::BigIntegerLiteral(val) => {
                self.visitor.visit_big_int(span, val)
            },
            Token::FloatLiteral(val) => {
                self.visitor.visit_float(span, val)
            },
            _ => unreachable!("unexpected constant: {}", tk)
        };
        /*
         * NOTE: `kind` is a 'u' for strings with a unicode prefix
         * Otherwise, it is `None`.
         * https://greentreesnakes.readthedocs.io/en/latest/nodes.html#Constant
         */
        Ok(self.visitor.visit_expr_constant(
            span, constant, kind
        ))
    }
    fn list_display(&mut self, tk: &SpannedToken<'a>) -> Result<V::Ident, ParseError> {
        debug_assert_eq!(tk.kind, Token::OpenBracket);
        match self.parser.peek().map(|tk| tk.kind) {
            Some(Token::CloseBracket) => {
                return Ok(self.visitor.visit_expr_list(
                    tk.span,
                    iter::empty(),
                    self.expression_context
                ))
            },
            _ => {},
        }
    }

}

