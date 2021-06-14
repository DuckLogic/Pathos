use crate::ast::constants::FloatLiteral;
use crate::lexer::{Token};
use crate::ast::{Ident, Span, Spanned};
use crate::ast::tree::*;

use super::parser::{SpannedToken, IParser, ParseError};
use super::{PythonParser};

use crate::vec;
use crate::alloc::{Allocator, AllocError, Vec};

struct PrefixParser<'src, 'a, 'p> {
    func: fn(
        parser: &mut PythonParser<'src, 'a, 'p>,
        token: &SpannedToken<'a>
    ) -> Result<Expr<'a>, ParseError>,
    // TODO: Is this needed?
    prec: ExprPrec
}
struct InfixParser<'src, 'a, 'p> {
    func: fn(
        parser: &mut PythonParser<'src, 'a, 'p>,
        left: Expr<'a>,
        token: &SpannedToken<'a>
    ) -> Result<Expr<'a>, ParseError>,
    prec: ExprPrec
}

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

impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    pub fn expression(&mut self) -> Result<Expr<'a>, ParseError> {
        self.parse_prec(ExprPrec::Atom)
    }
    /// A pratt parser for python expressions
    fn parse_prec(&mut self, prec: ExprPrec) -> Result<Expr<'a>, ParseError>{
        let token = self.parser.peek_tk();
        let mut left = match (token, token.as_ref().map(|tk| &tk.kind)
            .and_then(Self::prefix_parser)) {
            (Some(tk), Some(parser)) => {
                dbg!(self.parser.skip()?);
                (parser.func)(&mut *self, &tk)?
            },
            _ => {
                return Err(self.parser.unexpected(&"an expression"))
            }
        };
        loop {
            let next_token = match self.parser.peek_tk() {
                Some(tk) => tk, 
                None => return Ok(left)
            };
            match Self::infix_parser(&next_token.kind) {
                Some(parser) => {
                    if parser.prec >= prec {
                        return Ok(left)
                    }
                    self.parser.skip()?;
                    left = (parser.func)(self, left, &next_token)?
                },
                None => return Ok(left) // just give left
            }
        }
    }
    fn prefix_parser(token: &Token<'a>) -> Option<PrefixParser<'src, 'a, 'p>> {
        Some(match *token {
            Token::Ident(_) => PrefixParser {
                func: Self::name,
                prec: ExprPrec::Atom
            },
            Token::True | Token::False |
            Token::None | Token::StringLiteral(_) |
            Token::IntegerLiteral(_) |
            Token::BigIntegerLiteral(_) |
            Token::FloatLiteral(_) => PrefixParser {
                func: Self::constant,
                prec: ExprPrec::Atom
            },
            /*
             * For constructing a list, a set or a dictionary
             * Python provides special syntax called “displays”,
             * each of them in two flavors:
             * - either the container contents are listed explicitly, or
             * - they are computed via a set of looping and filtering
             *   instructions, called a comprehension.
             */
            Token::OpenBracket => PrefixParser {
                func: Self::list_display,
                prec: ExprPrec::Atom
            },

            Token::OpenParen => PrefixParser {
                func: Self::parentheses,
                prec: ExprPrec::Atom
            },
            Token::OpenBrace => PrefixParser {
                func: Self::dict_display,
                prec: ExprPrec::Atom
            },
            _ => return None
        })
    }
    fn infix_parser(token: &Token<'a>) -> Option<InfixParser<'src, 'a, 'p>> {
        Some(match *token {
            Token::Plus | Token::Minus | 
            Token::Star | Token::At |
            Token::Slash | Token::Percent |
            Token::DoubleStar | Token::LeftShift |
            Token::RightShift | Token::BitwiseOr |
            Token::BitwiseXor | Token::Ampersand |
            Token::DoubleSlash => InfixParser {
                func: Self::binary_op,
                prec: Operator::from_token(token)
                    .unwrap().precedence()
            },
            _ => return None
        })
    }
    fn binary_op(&mut self, left: Expr<'a>, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let op = match Operator::from_token(&tk.kind) {
            Some(it) => it,
            _ => unreachable!(),
        };
        let right = self.expression()?;
        let span = Span {
            start: left.span().start,
            end: right.span().end
        };
        Ok(&*self.arena.alloc(ExprKind::BinOp {
            left, span, op, right
        })?)
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

    fn parentheses(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        self.collection(tk, CollectionType::Tuple)
    }
    fn dict_display(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        self.collection(tk, CollectionType::Dict)
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
        /*
         * NOTE: Sets are never the default.
         * We always start parsing assuming we're
         * dealing with a dict, then switch to
         * a set later depending on the future syntax.
         */
        assert!(!collection_type.is_set());
        /*
         * NOTE: Tuples need special handling.
         * because they are created by the *COMMA*
         * operator and not by parentheses.
         *
         * However in our parser, they are
         * essentially implemented both ways.
         * We start out parsing a `(expr...` as a 'tuple'.
         * However, if there is only one element
         * and a closing paren without any comma
         * then it's clear the parens are for grouping only.
         *
         * NOTE: We still need to handle tuples created
         * without the parens, those that are using
         * the comma operator only (like `a, b = c, d`)
         */
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
            let mut elements: Vec<(Expr<'a>, Expr<'a>)> = vec![in self.arena; (first, fisrt_value.unwrap())]?;
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
                elements.push(val?)?;
            }
            let end = self.parser.expect(Token::CloseBrace)?.span.end;
            Ok(&*self.arena.alloc(ExprKind::Dict {
                span: Span { start, end },
                elements: elements.into_slice()
            })?)
        } else if collection_type.is_tuple() && self.parser.peek() == Some(Token::CloseParen) {
            self.parser.expect(Token::CloseParen)?;
            /*
             * These parentheses function for
             * grouping purposes only.
             * See note above.
             */
            Ok(first)
        } else {
            let mut elements = vec![in self.arena; first]?;
            for val in self.parse_terminated(
                Token::Comma,
                collection_type.closing_token(),
                |parser| parser.expression()
            ) {
                elements.push(val?)?;
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
    fn is_tuple(self) -> bool {
        matches!(self, CollectionType::Tuple)
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

#[cfg(test)]
mod test {
    use crate::alloc::{Allocator, AllocError};
    use bumpalo::Bump;
    use pretty_assertions::assert_eq;
    use crate::ast::tree::{Expr, ExprKind, Operator};
    use crate::ParseMode;
    use crate::ast::{Span, Constant};
    use crate::ast::constants::ConstantPool;
    use crate::{ident, expr};
    use std::error::Error;
    use std::backtrace::Backtrace;

    struct TestContext<'a, 'p> {
        arena: &'a Allocator,
        pool: &'p mut ConstantPool<'a>
    }
    impl<'a, 'p> TestContext<'a, 'p> {
        fn int(&mut self, i: i64) -> Expr<'a> {
            let val = self.pool.int(DUMMY, i).unwrap();
            self.constant(val)
        }
        fn constant(&self, value: Constant<'a>) -> Expr<'a> {
            self.arena.alloc(ExprKind::Constant {
                span: DUMMY,
                kind: None,
                value
            }).unwrap()
        }
    }
    fn test_expr(s: &str, expected: impl for<'a, 'p> FnOnce(&mut TestContext<'a, 'p>) -> Result<Expr<'a>, AllocError>) {
        let arena = Allocator::new(Bump::new());
        let mut pool = ConstantPool::new(&arena);
        let expected = {
            let mut ctx = TestContext { arena: &arena, pool: &mut pool };
            expected(&mut ctx).unwrap()
        };
        let actual = crate::parse(&arena, s, ParseMode::Expression, &mut pool)
            .unwrap_or_else(|e| panic!(
                "Failed to parse: {}\n\tBacktrace:\n{}", e,
                e.backtrace().unwrap_or(&Backtrace::disabled())
            ))
            .as_expression().unwrap();
        assert_eq!(expected, actual);
    }
    const DUMMY: Span = Span::dummy();
    #[test]
    fn literals() {
        test_expr("5", |ctx| Ok(ctx.int(5)));
        test_expr("5 + 5", |ctx| Ok(expr!(ctx, Expr::BinOp {
            span: DUMMY, left: ctx.int(5), op: Operator::Add, right: ctx.int(5)
        })))
    }
}
