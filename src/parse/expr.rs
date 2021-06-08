use std::iter;

use crate::lexer::{Token};
use crate::ast::{Span};
use crate::ast::tree::*;

use super::parser::{SpannedToken, Parser, IParser, ParseError};
use super::{PythonParser};

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

impl<'src, 'a> PythonParser<'src, 'a> {
    pub fn expression(&mut self) -> Result<Expr<'a>, ParseError> {
        self.parse_prec(ExprPrec::Atom)
    }
    /// A pratt parser for python expressions
    fn parse_prec(&mut self, prec: ExprPrec) -> Result<Expr<'a>, ParseError>{
        let token = self.parser.peek_tk();
        let left = match (token, token.as_ref().map(|tk| &tk.kind)
            .and_then(Self::prefix_parser)) {
            (Some(tk), Some(func)) => {
                self.parser.skip()?;
                func(&mut *self, &tk)?
            },
            _ => {
                return Err(self.parser.unexpected(&"an expression"))
            }
        };
        let next_token = match self.parser.peek_tk() {
            Some(tk) => tk, 
            None => return Ok(left)
        };
        match Self::infix_parser(&next_token.kind) {
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
            Token::OpenBracket => Self::list_display,
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
                )
            }, 
            _ => unreachable!()
        };
        Ok(self.visitor.visit_expr_name(
            tk.span,
            ident,
            self.expression_context
        ))
    }
    fn constant(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
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
                self.visitor.visit_string(span, *lit)
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
            _ => unreachable!("unexpected constant: {:?}", tk)
        };
        /*
         * NOTE: `kind` is a 'u' for strings with a unicode prefix
         * Otherwise, it is `None`.
         * https://greentreesnakes.readthedocs.io/en/latest/nodes.html#Constant
         */
        Ok(Expr::Constant {
            span, 
        })
    }

    fn parse_comprehension(
        &mut self,
    ) -> Result<V::Comprehension, ParseError> {
        todo!()
    }
    fn list_display(&mut self, tk: &SpannedToken<'a>) -> Result<V::Expr, ParseError> {
        self.collection(tk, CollectionType::List)
    }
    fn collection(
        &mut self,
        start_token: &SpannedToken<'a>,
        mut collection_type: CollectionType
    ) -> Result<V::Expr, ParseError> {
        debug_assert_eq!(
            start_token.kind,
            collection_type.opening_token()
        );
        // NOTE: Sets are never the default
        assert!(!collection_type.is_set());
        let start = start_token.span.start;
        if self.parser.peek() == Some(collection_type.closing_token()) {
            let end = self.parser.current_span().end;
            self.parser.skip()?;
            if collection_type.is_dict() {
                return Ok(self.visitor.visit_expr_dict(
                    Span { start, end },
                    iter::empty(),
                    iter::empty(),
                ));
            } else {
                return Ok(collection_type.visit_simple(
                    self.visitor,
                    Span { start, end },
                    iter::empty(),
                    self.expression_context
                ));
            }
        }
        let first = self.expression()?;
        let fisrt_value = if collection_type.is_dict() {
            /*
             * Check if it's actually a dict.
             * It's also possible it's just a set
             */
            if self.parser.peek() == Some(Token::Colon) {
                self.parser.skip()?;
                // It's a dict alright
                Some(self.expression()?)
            } else {
                // Actually a set in disguise
                collection_type = CollectionType::Set;
                None
            }
        } else {
            None
        };
        if self.parser.peek() == Some(Token::For) {
            let mut comprehensions = Vec::with_capacity(1);
            while self.parser.peek() == Some(Token::For) {
                comprehensions.push(self.parse_comprehension()?);
            }
            let end = self.parser.expect(Token::CloseBracket)?.span.end;
            if collection_type.is_dict() {
                Ok(self.visitor.visit_expr_dict_comp(
                    Span { start, end },
                    first, fisrt_value.unwrap(),
                    comprehensions.into_iter()
                ))
            } else {
                Ok(collection_type.visit_comprehension(
                    self.visitor, Span { start, end },
                    first, comprehensions.into_iter()
                ))
            }
        } else {
            assert!(
                !collection_type.is_dict(),
                "TODO: Dictionary literals"
            );
            let mut elements = vec![first];
            for val in self.parse_terminated(
                Token::Comma,
                Token::CloseBracket,
                |parser| parser.expression()
            ) {
                elements.push(val?);
            }
            let end = self.parser.expect(Token::CloseBracket)?.span.end;
            Ok(collection_type.visit_simple(
                self.visitor,
                Span { start, end },
                elements.into_iter(),
                self.expression_context
            ))
        }
    }

}

#[derive(Copy, Clone, Debug)]
enum CollectionType {
    List,
    Tuple,
    Set,
    Dict
}
impl CollectionType {
    #[inline]
    fn visit_simple<'a>(
        self,
        visitor: &mut V,
        span: Span,
        elements: impl Iterator<Item=V::Expr>,
        ctx: ExprContext
    ) -> V::Expr {
        match self {
            CollectionType::List => {
                visitor.visit_expr_list(span, elements, ctx)
            },
            CollectionType::Set => {
                visitor.visit_expr_set(span, elements)
            },
            CollectionType::Tuple => {
                visitor.visit_expr_tuple(span, elements, ctx)
            },
            CollectionType::Dict => unreachable!()
        }
    }
    #[inline]
    fn visit_comprehension<'a>(
        self,
        visitor: &mut V,
        span: Span,
        element: V::Expr,
        comprehensions: impl Iterator<Item=V::Comprehension>,
    ) -> V::Expr {
        match self {
            CollectionType::List => {
                visitor.visit_expr_list_comp(span, element, comprehensions)
            },
            CollectionType::Tuple => {
                // NOTE: (a for x in y) is actually
                // a generator, not a 'tuple comprehension'
                visitor.visit_expr_generator_exp(span, element, comprehensions)
            },
            CollectionType::Set => {
                visitor.visit_expr_set_comp(span, element, comprehensions)
            },
            CollectionType::Dict => unreachable!()
        }
    }
    #[inline]
    fn is_set(self) -> bool {
        matches!(self, CollectionType::Set)
    }
    #[inline]
    fn is_dict(self) -> bool {
        matches!(self, CollectionType::Dict)
    }
    #[inline]
    fn opening_token(self) -> Token<'static> {
        match self {
            CollectionType::List => Token::OpenBracket,
            CollectionType::Tuple => Token::OpenParen,
            CollectionType::Set |
            CollectionType::Dict => Token::OpenBrace,
        }
    }
    #[inline]
    fn closing_token(self) -> Token<'static> {
        match self {
            CollectionType::List => Token::CloseBracket,
            CollectionType::Tuple => Token::CloseParen,
            CollectionType::Set |
            CollectionType::Dict => Token::CloseBrace,            
        }
    }
}

