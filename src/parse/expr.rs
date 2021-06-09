use crate::ast::constants::FloatLiteral;
use crate::lexer::{Token};
use crate::ast::{Span, Ident};
use crate::ast::tree::*;

use super::parser::{SpannedToken, IParser, ParseError};
use super::{PythonParser};

use crate::vec;
use crate::alloc::{Allocator, AllocError, Vec};

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
pub enum ExprPrec {
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
        -> Option<fn(&mut Self, &SpannedToken<'a>) -> Result<Expr<'a>, ParseError>> {
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
        -> Option<fn(&mut Self, Expr<'a>, &SpannedToken<'a>) -> Result<Expr<'a>, ParseError>> {
        Some(match *token {
            Token::Plus | Token::Minus | 
            Token::Star | Token::At |
            Token::Div | Token::Percent |
            Token::DoubleStar | Token::LeftShift |
            Token::RightShift | Token::BitwiseOr |
            Token::BitwiseXor | Token::BitwiseAnd |
            Token::DoubleSlash => Self::binary_op,
            _ => return None
        })
    }
    fn name(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let ident = match **tk {
            Token::Ident(inner) => inner, 
            _ => unreachable!()
        };
        let ident = *self.arena.alloc(Ident::from_raw(tk.span, ident)?)?;
        Ok(&*self.arena.alloc(ExprKind::Name {
            span: tk.span, id: ident,
            ctx: self.expression_context
        })?)
    }
    fn constant(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let span = tk.span;
        let mut kind = None;
        let constant = match tk.kind {
            Token::True => self.pool.bool(span, true)?,
            Token::False => self.pool.bool(span, false)?,
            Token::None => self.pool.none(span)?,
            Token::StringLiteral(lit) => {
                match lit.style.prefix {
                    Some(crate::ast::constants::StringPrefix::Unicode) => {
                        kind = Some("u");
                    },
                    _ => {}
                }
                self.pool.string(span, *lit)?
            },
            Token::IntegerLiteral(val) => {
                self.pool.int(span, val)?
            },
            Token::BigIntegerLiteral(val) => {
                self.pool.big_int(span, val)?
            },
            Token::FloatLiteral(val) => {
                self.pool.float(span, FloatLiteral::new(val).unwrap())?
            },
            _ => unreachable!("unexpected constant: {:?}", tk)
        };
        /*
         * NOTE: `kind` is a 'u' for strings with a unicode prefix
         * Otherwise, it is `None`.
         * https://greentreesnakes.readthedocs.io/en/latest/nodes.html#Constant
         */
        Ok(&*self.arena.alloc(ExprKind::Constant {
            span, kind, value: constant
        })?)
    }

    fn parse_comprehension(
        &mut self,
    ) -> Result<Comprehension<'a>, ParseError> {
        todo!()
    }
    fn list_display(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        self.collection(tk, CollectionType::List)
    }
    fn collection(
        &mut self,
        start_token: &SpannedToken<'a>,
        mut collection_type: CollectionType
    ) -> Result<Expr<'a>, ParseError> {
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
                return Ok(&*self.arena.alloc(ExprKind::Dict {
                    span: Span { start, end },
                    elements: &[]
                })?);
            } else {
                return Ok(collection_type.create_simple(
                    self.arena,
                    Span { start, end },
                    &[],
                    self.expression_context
                )?);
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
            let mut comprehensions = Vec::with_capacity(self.arena, 1)?;
            while self.parser.peek() == Some(Token::For) {
                comprehensions.push(self.parse_comprehension()?)?;
            }
            let end = self.parser.expect(Token::CloseBracket)?.span.end;
            if collection_type.is_dict() {
                Ok(&*self.arena.alloc(ExprKind::DictComp {
                    span: Span { start, end },
                    key: first,
                    value: fisrt_value.unwrap(),
                    generators: comprehensions.into_slice()
                })?)
            } else {
                Ok(collection_type.create_comprehension(
                    self.arena, Span { start, end },
                    first, comprehensions.into_slice()
                )?)
            }
        } else if collection_type.is_dict() {
            let mut elements = vec![in self.arena; (first, fisrt_value.unwrap())]?;
            for val in self.parse_terminated(
                Token::Comma,
                Token::CloseBrace,
                |parser| {
                    let key = parser.expression()?;
                    parser.parser.expect(Token::Colon)?;
                    let value = parser.expression()?;
                    Ok((key, value))
                }
            ) {
                elements.push(val?);
            }
            let end = self.parser.expect(Token::CloseBrace)?.span.end;
            Ok(&*self.arena.alloc(ExprKind::Dict {
                span: Span { start, end },
                elements: elements.into_slice()
            })?)
        } else {
            let mut elements = vec![in self.arena; first]?;
            for val in self.parse_terminated(
                Token::Comma,
                collection_type.closing_token(),
                |parser| parser.expression()
            ) {
                elements.push(val?);
            }
            let end = self.parser.expect(collection_type.closing_token())?
                .span.end;
            Ok(collection_type.create_simple(
                self.arena,
                Span { start, end },
                elements.into_slice(),
                self.expression_context
            )?)
        }
    }
    fn binary_op(
        &mut self,
        start_token: &SpannedToken<'a>,
        left: &SpannedToken,
    )
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
    fn create_simple<'a>(
        self,
        arena: &'a Allocator,
        span: Span,
        elements: &'a [Expr<'a>],
        ctx: ExprContext
    ) -> Result<Expr<'a>, AllocError> {
        Ok(&*arena.alloc(match self {
            CollectionType::List => {
                ExprKind::List { span, elts: elements, ctx }
            },
            CollectionType::Set => {
                ExprKind::Set { span, elts: elements }
            },
            CollectionType::Tuple => {
                ExprKind::Tuple { span, elts: elements, ctx }
            },
            CollectionType::Dict => unreachable!()
        })?)
    }
    #[inline]
    fn create_comprehension<'a>(
        self,
        arena: &'a Allocator,
        span: Span,
        element: Expr<'a>,
        generators: &'a [Comprehension],
    ) -> Result<Expr<'a>, AllocError> {
        Ok(&*arena.alloc(match self {
            CollectionType::List => {
                ExprKind::ListComp { span, elt: element, generators }
            },
            CollectionType::Tuple => {
                // NOTE: (a for x in y) is actually
                // a generator, not a 'tuple comprehension'
                ExprKind::GeneratorExp { span, elt: element, generators }
            },
            CollectionType::Set => {
                ExprKind::SetComp { span, elt: element, generators }
            },
            CollectionType::Dict => unreachable!()
        })?)
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

