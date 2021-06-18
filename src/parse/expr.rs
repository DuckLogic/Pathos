//! A pratt parser for Python expressions
//!
//! Primarily based upon [Bob Nystorm's excellent blog post](http://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/)
//!
//! See also [this (C) pratt parser, implemented in crafting interpreters](http://craftinginterpreters.com/compiling-expressions.html#parsing-prefix-expressions)
//! See also [this example code](https://github.com/munificent/bantam)
use std::fmt::Display;
use std::ops::{Add, Sub};

use crate::alloc::{Allocator, AllocError, Vec};
use crate::ast::{Ident, Span, Spanned};
use crate::ast::constants::FloatLiteral;
use crate::ast::tree::*;
use crate::lexer::Token;
use crate::parse::errors::ParseError;
use crate::vec;

use super::parser::{IParser, SpannedToken};
use super::PythonParser;
use crate::parse::ArgumentParseOptions;
use crate::parse::parser::{EndFunc, ParseSeperated};

struct PrefixParser<'src, 'a, 'p> {
    func: fn(
        parser: &mut PythonParser<'src, 'a, 'p>,
        token: &SpannedToken<'a>
    ) -> Result<Expr<'a>, ParseError>,
    prec: ExprPrec
}
impl<'src, 'a, 'p> PrefixParser<'src, 'a, 'p> {
    #[inline]
    fn atom(func: fn(parser: &mut PythonParser<'src, 'a, 'p>, token: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError>) -> Self {
        PrefixParser { func, prec: ExprPrec::Atom }
    }
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
#[repr(u8)]
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
    const MIN_PREC: ExprPrec = ExprPrec::from_int(0).unwrap();
    /// The dummy expression precedence, used to parse target lists
    ///
    /// This is because target lists are currently implemented as expressions.
    const TARGET_PREC: ExprPrec = ExprPrec::BitwiseOr;
    const fn from_int(val: u8) -> Option<Self> {
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
impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    /// Parse an expression
    ///
    /// Parses anything regardless of precedence
    pub fn expression(&mut self) -> Result<Expr<'a>, ParseError> {
        self.parse_prec(ExprPrec::MIN_PREC)
    }
    /// Parse an expression, accepting anything that has at least the specified precedence (or binding power)
    ///
    /// For example, say you are the `+` parser and you are parsing the input `3 + 5 * 6`
    /// You would call parse_prec(+) with the `5 * 6` on your right.
    /// Since '*' has a higher precedence (more binding power), it would be fully consumed
    /// and you would get `3 + (5 * 6)`.
    ///
    /// On the other hand, say you were parsing the input `5 * 6 + 3`
    /// You would call parse_prec(*) with the `6 + 3` on your right.
    /// Since `+` has a lower precedence (less binding power) than the min_prec `+`,
    /// it would **NOT** be consumed.
    fn parse_prec(&mut self, min_prec: ExprPrec) -> Result<Expr<'a>, ParseError>{
        let token = self.parser.peek_tk();
        let mut left = match (token, token.as_ref().map(|tk| &tk.kind)
            .and_then(Self::prefix_parser)) {
            (Some(tk), Some(parser)) if parser.prec >= min_prec => {
                self.parser.skip()?;
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
                    if parser.prec <= min_prec {
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
        let atom = PrefixParser::atom;
        Some(match *token {
            Token::Ident(_) => atom(Self::name),
            Token::True | Token::False |
            Token::None | Token::StringLiteral(_) |
            Token::IntegerLiteral(_) |
            Token::BigIntegerLiteral(_) |
            Token::FloatLiteral(_) => atom(Self::constant),
            /*
             * For constructing a list, a set or a dictionary
             * Python provides special syntax called “displays”,
             * each of them in two flavors:
             * - either the container contents are listed explicitly, or
             * - they are computed via a set of looping and filtering
             *   instructions, called a comprehension.
             */
            Token::OpenBracket => atom(Self::list_display),
            Token::OpenParen => atom(Self::parentheses),
            Token::OpenBrace => atom(Self::dict_display),
            Token::BitwiseInvert | Token::Not |
            Token::Plus | Token::Minus => {
                PrefixParser {
                    func: Self::unary_op,
                    prec: ExprPrec::Unary
                }
            },
            Token::Lambda => {
                PrefixParser {
                    func: Self::lambda,
                    prec: ExprPrec::Lambda
                }
            }
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
            Token::Period => InfixParser {
                prec: ExprPrec::Call,
                func: Self::attr_reference
            },
            _ => return None
        })
    }
    /// Parse a yield expression (doesn't handle any wrapping parenthesis)
    ///
    /// This accepts an [EndFunc]
    /// to decide when parsing the yield should end.
    /// This is because the single-form of from can actually yield nothing, or yield a list.
    /// For example, both `yield` and `yield a, b` are valid.
    ///
    /// A yield expression can exist in any expression position,
    /// provided that it is wrapped in parenthesis `5 + (yield inner)`.
    ///
    /// However, in certain cases the parens can be omitted:
    /// 1. When the single element of an expression-statement
    /// 2. When it is the sole expression on the right hand side of an assignment statemenet.
    ///
    /// Therefore, we keep this function public to allow those two special-cases.
    ///
    /// See docs: <https://docs.python.org/3.10/reference/expressions.html#yield-expressions>
    pub fn parse_yield_expr(&mut self, mut end_func: impl EndFunc<'src, 'a>) -> Result<Expr<'a>, ParseError> {
        let original_keyword_span = self.parser.expect(Token::Yield)?.span;
        let start = original_keyword_span.start;
        if let Some(Token::From) = self.parser.peek() {
            self.parser.skip()?;
            // From expressions are easy. We require one and only one value
            let value = self.expression()?;
            Ok(&*self.arena.alloc(ExprKind::YieldFrom {
                span: Span { start, end: value.span().end },
                value
            })?)
        } else {
            /*
             * In a a regular yield statement, we may encounter multiple items or none at all.
             * First, check for an early end, then parse the first expression,
             * then, check if we have more.
             */
            let value = if !end_func.should_end(&self.parser) {
                let first = self.expression()?;
                if end_func.should_end(&self.parser) {
                    // Looks like we only have a single expression
                    Some(first)
                } else if let Some(Token::Comma) = self.parser.peek() {
                    /*
                     * We have multiple expressions (implicitly packing into a tuple).
                     * Eat the comma, then delegate to parse seperated
                     */
                    let mut elements = vec![in self.arena; first]?;
                    self.parser.expect(Token::Comma)?;
                    let mut iter = ParseSeperated::new(
                        self, |p| p.expression(),
                        Token::Comma, end_func, true
                    );
                    while let Some(e) = iter.next().transpose()? {
                        elements.push(e)?;
                    }
                    Some(&*self.arena.alloc(ExprKind::Tuple {
                        span: Span { start: first.span().start, end: elements.last().unwrap().span().end },
                        ctx: self.expression_context,
                        elts: elements.into_slice()
                    })?)
                } else {
                    return Err(self.parser.unexpected(&format_args!("a comma or {}", end_func.description())))
                }
            } else { None };
            let end = match value {
                Some(val) => val.span().end,
                None => original_keyword_span.end
            };
            Ok(&*self.arena.alloc(ExprKind::Yield { value, span: Span { start, end } })?)
        }
    }
    fn attr_reference(&mut self, left: Expr<'a>, tk: &SpannedToken) -> Result<Expr<'a>, ParseError> {
        assert_eq!(tk.kind, Token::Period);
        let name = self.parse_ident()?;
        Ok(&*self.arena.alloc(ExprKind::Attribute {
            value: left,
            ctx: self.expression_context,
            span: Span { start: left.span().start, end: name.span().end },
            attr: name
        })?)
    }
    fn lambda(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let start = tk.span.start;
        assert_eq!(tk.kind, Token::Lambda);
        let args = self.parse_argument_declarations(Token::Colon, ArgumentParseOptions {
            allow_type_annotations: false
        })?;
        self.parser.expect(Token::Colon)?;
        let body = self.expression()?;
        Ok(&*self.arena.alloc(ExprKind::Lambda {
            span: Span { start, end: body.span().end }, args, body
        })?)
    }
    fn binary_op(&mut self, left: Expr<'a>, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let op = match Operator::from_token(&tk.kind) {
            Some(it) => it,
            _ => unreachable!(),
        };
        let mut min_prec = op.precedence();
        if op.is_right_associative() {
            /*
             * See here:
             * https://github.com/munificent/bantam/blob/master/src/com/stuffwithstuff/bantam/parselets/BinaryOperatorParselet.java#L20-L23
             *
             * If we are right associative (**), then we decrease the minimum precedence to allow
             * a parser with the exact same precedence to appear on our right.
             * For example, if we're at the first addition sign in "1+2+3", then we call parse_prec(ExprPrec::Add)
             * we would normally refuse to recurse into parsing the (2+3) on the right hand size,
             * because next_token.prec <= min_prec. This gives us (1+2), then we go on to parse the `3`,
             * giving us a result ((1+2)+3) and the left associativity we desire.
             * However, if we were to call parse_prec(ExprPrec::Add - 1),
             * we **would** recurse into the (2+3) on the right side,
             * giving us (1+(2+3)). For exponentiation `**`,
             * we actually want this right associativity, so we do the subtraction here.
             */
            min_prec = min_prec - 1;
        }
        let right = self.parse_prec(min_prec)?;
        let span = Span {
            start: left.span().start,
            end: right.span().end
        };
        Ok(&*self.arena.alloc(ExprKind::BinOp {
            span, left, right, op
        })?)
    }
    fn unary_op(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let op = match UnaryOp::from_token(&tk.kind) {
            Some(op) => op,
            None => unreachable!()
        };
        let right = self.parse_prec(op.precedence())?;
        let span = Span {
            start: tk.span.start,
            end: right.span().end
        };
        Ok(&*self.arena.alloc(ExprKind::UnaryOp {
            operand: right, span, op
        })?)
    }
    fn name(&mut self, tk: &SpannedToken<'a>) -> Result<Expr<'a>, ParseError> {
        let symbol = match **tk {
            Token::Ident(inner) => inner, 
            _ => unreachable!()
        };
        let ident = Ident { symbol, span: tk.span };
        Ok(&*self.arena.alloc(ExprKind::Name { span: ident.span, ctx: self.expression_context, id: ident })?)
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
    #[inline]
    fn peek_is_comprehension(&self) -> bool {
        matches!(self.parser.peek(), Some(Token::Async) | Some(Token::For))
    }
    fn parse_single_comprehension(&mut self) -> Result<Comprehension<'a>, ParseError> {
        let is_async = if let Some(Token::Async) = self.parser.peek() {
            self.parser.skip()?;
            true
        } else { false };
        self.parser.expect(Token::For)?;
        let target = self.parse_target_list(
            &mut |parser| parser.parser.peek() == Some(Token::In),
            &"'in'"
        )?;
        self.parser.expect(Token::In)?;
        // For the precedence of an iterator, we can have anything with higher precedence than a conditional
        let iter = self.parse_prec(ExprPrec::Conditional + 1)?;
        let mut ifs = Vec::new(self.arena);
        while let Some(Token::If) = self.parser.peek() {
            self.parser.skip()?; // Om nom nom
            /*
             * For the precedence of the condition in a comprehension expression,
             * we can have anything higher than a conditional expression (since that would be nonsense).
             * In other words it is illegal to have [e for e in l if (1 if e else 5)]
             * without having the parentheses.
             * NOTE: It is legal to parse these as two seperate if conditions.
             */
            ifs.push(self.parse_prec(ExprPrec::Conditional + 1)?)?;
        }
        Ok(Comprehension { iter, target, is_async, ifs: ifs.into_slice() })
    }
    /// Parses a list of target expressions,
    /// until we see the ending identified by the specified closure.
    ///
    /// The closure is necessary to avoid backtracking.
    ///
    /// Implicitly converts a list to a tuple expression (with the appropriate [ExprContext])
    fn parse_target_list(&mut self, should_end: &mut dyn FnMut(&Self) -> bool, expected_ending: &dyn Display) -> Result<Expr<'a>, ParseError> {
        let first = self.parse_target()?;
        if let Some(Token::Comma) = self.parser.peek() {
            let start = first.span().start;
            let mut v = vec![in self.arena; first]?;
            while let Some(Token::Comma) = self.parser.peek() {
                self.parser.skip()?;
                if should_end(self) { break; }
                v.push(self.parse_target().map_err(|err| {
                    err.with_expected_msg(format_args!("either an expression or {}", expected_ending))
                })?)?;
            }
            let end = v.last().unwrap().span().end;
            Ok(&*self.arena.alloc(ExprKind::Tuple {
                span: Span { start, end }, ctx: self.expression_context,
                elts: v.into_slice()
            })?)
        } else {
            Ok(first)
        }
    }
    /// Parses a target expression
    ///
    /// This corresponds to "target" in the grammar (see the section on assignments):
    /// https://docs.python.org/3.10/reference/simple_stmts.html#assignment-statements
    fn parse_target(&mut self) -> Result<Expr<'a>, ParseError> {
        self.parse_prec(ExprPrec::TARGET_PREC)
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
         * because they are technically created by the *COMMA*
         * operator and not by parentheses (at least according to the reference).
         *
         * However, the places where an un-parenthesized tuple
         * can become an expression are somewhat restricted,
         * because comma-seperated lists can mean different things
         * in different contexts.
         * Therefore we just sprinkle some implicit tuple creation
         * around without technically defining a 'comma operator'.
         *
         * Parenthesizes on the other hand, *do* create a tuple
         * as long as there is more than one element inside.
         * We start out parsing a `(expr ...` as a 'tuple'.
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
        if collection_type.is_tuple() && self.parser.peek() == Some(Token::Yield) {
            /*
             * NOTE: Yield tokens need to be special-cased,
             * because they are only allowed inside parens (aside from a few exceptions).
             */
            // This is actually a yield expression
            let yield_expr = self.parse_yield_expr(Token::CloseParen)?;
            self.parser.expect(Token::CloseParen)?;
            return Ok(yield_expr)
        }
        let first = self.expression()?;
        let first_value = if collection_type.is_dict() {
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
        if self.peek_is_comprehension() {
            // NOTE: Comprehensions can be nested like so:
            // [e1 * e2 for e1 in l1 for e2 in l2]
            // This gives the product (or combinations) of l1 & l2
            let mut comprehensions = Vec::with_capacity(self.arena, 1)?;
            while self.peek_is_comprehension() {
                comprehensions.push(self.parse_single_comprehension()?)?;
            }
            let end = self.parser.expect(collection_type.closing_token())?.span.end;
            if collection_type.is_dict() {
                Ok(&*self.arena.alloc(ExprKind::DictComp {
                    span: Span { start, end },
                    key: first,
                    value: first_value.unwrap(),
                    generators: comprehensions.into_slice()
                })?)
            } else {
                Ok(collection_type.create_comprehension(
                    self.arena, Span { start, end },
                    first, comprehensions.into_slice()
                )?)
            }
        } else if collection_type.is_dict() {
            /*
             * Two possibilities here: ',' or '}'
             *
             * See comment in the final 'else' clause for more detailed reasoning.
             */
            match self.parser.peek() {
                Some(Token::Comma) => {
                    self.parser.skip()?; // Consume comma, making us ready for ParseSeperated
                },
                Some(Token::CloseBrace) => {
                    // Ignore this, ParseSeperated will handle it for us
                }
                _ => {
                    return Err(self.parser.unexpected(&"Either a comprehension, a comma, or '}'"));
                }
            }
            let mut elements: Vec<(Expr<'a>, Expr<'a>)> = vec![in self.arena; (first, first_value.unwrap())]?;
            for val in self.parse_terminated(
                Token::Comma, Token::CloseBrace,
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
             * See note above on the 'comma operator'.
             */
            Ok(first)
        } else {
            /*
             * We've already consumed the first expression,
             * and we've ruled out the possibility of an generator.
             * Therefore, the only remaining possibilities are a comma,
             * or an immediate closing token.
             * If we have a comma, consume it and then delegate to ParseSeperated.
             * If we have a closing token, ignore it, and let ParseSeperated handle it.
             *
             * If we encounter an error, we should of all three possibilities,
             * including the option of a generator/comprehension.
             */
            match self.parser.peek() {
                Some(Token::Comma) => {
                    self.parser.skip()?; // Consume comma, making us ready for ParseSeperated
                },
                Some(closing) if closing == collection_type.closing_token() => {
                    // Ignore this, ParseSeperated will handle it for us
                }
                _ => {
                    return Err(self.parser.unexpected(&format_args!(
                        "Either a comprehension, a comma, or '{}'",
                        collection_type.closing_token()
                    )));
                }
            }
            let mut elements = vec![in self.arena; first]?;
            for val in self.parse_terminated(
                Token::Comma,
                collection_type.closing_token(),
                |parser| parser.expression()
            ) {
                elements.push(val?)?;
            }
            let end = self.parser.expect(collection_type.closing_token())?.span.end;
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
    use std::backtrace::Backtrace;
    use std::cell::RefCell;
    use std::error::Error;

    use bumpalo::Bump;
    use pretty_assertions::assert_eq;

    use crate::alloc::{Allocator, AllocError};
    use crate::ast::AstBuilder;
    use crate::ast::constants::ConstantPool;
    use crate::ast::ident::SymbolTable;
    use crate::ast::tree::*;
    use crate::parse::ExprPrec;
    use crate::parse::test::{DUMMY, TestContext};
    use crate::ParseMode;

    macro_rules! vec {
        ($ctx:expr) => (vec![$ctx,]);
        ($ctx:expr, $($elements:expr),*) => {
            &*crate::vec![in $ctx.arena; $($elements),*].unwrap().into_slice()
        };
    }
    #[track_caller]
    fn test_expr(s: &str, expected: impl for<'a, 'p> FnOnce(&mut TestContext<'a, 'p>) -> Result<Expr<'a>, AllocError>) {
        let arena = Allocator::new(Bump::new());
        let mut pool = ConstantPool::new(&arena);
        let mut symbol_table = SymbolTable::new(&arena);
        let expected = {
            let mut ctx = TestContext { arena: &arena, pool: RefCell::new(&mut pool), builder: AstBuilder { arena: &arena },
                symbol_table: RefCell::new(&mut symbol_table) };
            expected(&mut ctx).unwrap()
        };
        let actual = crate::parse(&arena, s, ParseMode::Expression, &mut pool, &mut symbol_table)
            .unwrap_or_else(|e| panic!(
                "Failed to parse: {}\n\tBacktrace:\n{}", e,
                e.backtrace().unwrap_or(&Backtrace::disabled())
            ))
            .as_expression().unwrap();
        assert_eq!(expected, actual);
    }
    #[test]
    fn literals() {
        test_expr("5", |ctx| Ok(ctx.int(5)));
        test_expr("5 + 5", |ctx| Ok(ctx.bin_op(
            ctx.int(5), Operator::Add, ctx.int(5)
        )));
    }
    #[test]
    fn arith_prec() {
        assert!(ExprPrec::Term < ExprPrec::Factor); // + has less binding power than '*'
        test_expr("5 + 3 * 6", |ctx| Ok(ctx.bin_op(
            ctx.int(5), Operator::Add,
            ctx.bin_op(ctx.int(3), Operator::Mult, ctx.int(6))
        )));
        test_expr("3 * 6 + 5", |ctx| Ok(ctx.bin_op(
            ctx.bin_op(ctx.int(3), Operator::Mult, ctx.int(6)),
            Operator::Add, ctx.int(5)
        )));
        // NOTE: This parses as -(1 ** 3)
        test_expr("-1**3", |ctx| Ok(ctx.expr(ExprKind::UnaryOp {
            span: DUMMY, op: UnaryOp::USub, operand: ctx.bin_op(
                ctx.int(1), Operator::Pow, ctx.int(3)
            )
        })));
    }
    #[test]
    fn attribute_reference() {
        test_expr("a.b", |ctx| Ok(ctx.attr_ref(ctx.name("a"), "b")));
        test_expr("a.b.c", |ctx| Ok(ctx.attr_ref(
            ctx.attr_ref(ctx.name("a"), "b"),
            "c"
        )));
    }
    #[test]
    fn test_arith_associativity() {
        /*
         * Quick lesson on associativity:
         * Normal arithmetic obeys the "associative property" (from Algebra 1)
         * so that (1 + (2 + 3)) == (1 + (2 + 3)). It doesn't matter how you group
         * or order your addition, the result is the same.
         * However, because of operator overloading, this may not be the case in Python.
         * Almost all python operators are parsed using whats called "left associativity",
         * so that (in the absence of explicit grouping) "1 + 2 + 3 + 4"
         * parses as "(((1 + 2) + 3) + 4)".
         * The alternative is what is called "right associativity". That would parse the
         * same expression as "(1 + (2 + (3 + 4)))".
         * In Python, the only place this "right associativity" is used is the power operator (**).
         * "1**2**3" parses as "(1**(2**3))". Everything else parses with left
         * associativity.
         */
        // Should parse as ((1+2)+3) <left associative>
        test_expr("1+2+3", |ctx| Ok(ctx.bin_op(
            ctx.bin_op(ctx.int(1), Operator::Add, ctx.int(2)),
            Operator::Add, ctx.int(3)
        )));
        /*
         * Should parse as (((1*2)*3)/4) <left associative>
         * NOTE: Multiplication and division *should* have the same precedence
         */
        test_expr("1*2*3/4", |ctx| Ok(ctx.bin_op(
            ctx.bin_op(
                ctx.bin_op(ctx.int(1), Operator::Mult, ctx.int(2)),
                Operator::Mult, ctx.int(3)
            ),
            Operator::Div, ctx.int(4)
        )));

        /*
         * Exponentiation is the only exception to the left associativity.
         * This parse as (1**(2**3)) <right associative>
         */
        test_expr("1**2**3", |ctx| Ok(ctx.bin_op(
            ctx.int(1), Operator::Pow, ctx.bin_op(
                ctx.int(2),  Operator::Pow, ctx.int(3)
            )
        )));
    }
    #[test]
    fn collections() {
        test_expr("[1, 2, 3]", |ctx| Ok(ctx.expr(ExprKind::List {
            span: DUMMY, elts: vec!(ctx, ctx.int(1), ctx.int(2), ctx.int(3)),
            ctx: ExprContext::Load
        })));
        let list_of_one: for<'a, 'p> fn(&mut TestContext<'a, 'p>) -> Result<Expr<'a>, _> = |ctx| {
            Ok(ctx.expr(ExprKind::List {
                span: DUMMY,
                elts: vec!(ctx, ctx.int(1)),
                ctx: ExprContext::Load
            }))
        };
        test_expr("[1]", list_of_one);
        test_expr("[1,]", list_of_one);
        test_expr("[1,2]", |ctx| Ok(ctx.expr(ExprKind::List {
            span: DUMMY, elts: vec![ctx, ctx.int(1), ctx.int(2)],
            ctx: ExprContext::Load
        })));
        test_expr("[]", |ctx| Ok(ctx.expr(ExprKind::List {
            span: DUMMY, elts: &[],
            ctx: ExprContext::Load
        })));
        test_expr("[1, 2, 4,]", |ctx| Ok(ctx.expr(ExprKind::List {
            span: DUMMY, elts: vec![ctx, ctx.int(1), ctx.int(2), ctx.int(4)],
            ctx: ExprContext::Load
        })));
        test_expr("(1, foo)", |ctx| Ok(ctx.expr(ExprKind::Tuple {
            span: DUMMY, elts: vec![ctx, ctx.int(1), ctx.name("foo")],
            ctx: ExprContext::Load
        })));
        test_expr("(1, foo, toad,)", |ctx| Ok(ctx.expr(ExprKind::Tuple {
            span: DUMMY, elts: vec![ctx, ctx.int(1), ctx.name("foo"), ctx.name("toad")],
            ctx: ExprContext::Load
        })));
    }
    #[test]
    fn comprehensions() {
        test_expr("[e for e in l]", |ctx| Ok(ctx.expr(ExprKind::ListComp {
            span: DUMMY, elt: ctx.name("e"), generators: vec!(ctx, Comprehension {
                target: ctx.name("e"), iter: ctx.name("l"), ifs: &[], is_async: false
            })
        })));
        test_expr("[e1 * e2 for e1 in l1 for e2 in l2]", |ctx| Ok(ctx.expr(ExprKind::ListComp {
            span: DUMMY, elt: ctx.bin_op(
                ctx.name("e1"),
                Operator::Mult,
                ctx.name("e2")
            ),
            generators: vec![ctx, Comprehension {
                target: ctx.name("e1"), iter: ctx.name("l1"), ifs: &[], is_async: false
            }, Comprehension {
                target: ctx.name("e2"), iter: ctx.name("l2"), ifs: &[], is_async: false
            }]
        })));
        test_expr("[e async for e in l if e + 3]", |ctx| Ok(ctx.expr(ExprKind::ListComp {
            span: DUMMY, elt: ctx.name("e"),
            generators: vec![ctx, Comprehension {
                target: ctx.name("e"), iter: ctx.name("l"),
                ifs: vec![ctx, ctx.bin_op(
                    ctx.name("e"),
                    Operator::Add,
                    ctx.int(3)
                )],
                is_async: true
            }]
        })));
        test_expr("(i for i in gen)", |ctx| Ok(ctx.expr(ExprKind::GeneratorExp {
            span: DUMMY, elt: ctx.name("i"),
            generators: vec![ctx, Comprehension {
                target: ctx.name("i"), iter: ctx.name("gen"),
                ifs: &[], is_async: false
            }]
        })));
    }
    #[test]
    fn dictionaries() {
        test_expr("{}", |ctx| Ok(ctx.expr(ExprKind::Dict {
            span: DUMMY, elements: &[]
        })));
        test_expr("{key: value}", |ctx| Ok(ctx.expr(ExprKind::Dict {
            span: DUMMY,elements: vec!(ctx, (ctx.name("key"), ctx.name("value")))
        })));
        test_expr("{key: value,}", |ctx| Ok(ctx.expr(ExprKind::Dict {
            span: DUMMY, elements: vec!(ctx, (ctx.name("key"), ctx.name("value")))
        })));
        test_expr("{a: b, c: d}", |ctx| Ok(ctx.expr(ExprKind::Dict {
            span: DUMMY, elements: vec!(ctx, (ctx.name("a"), ctx.name("b")), (ctx.name("c"), ctx.name("d")))
        })));
    }
    #[test]
    fn lambdas() {
        test_expr("lambda a, b: a + b", |ctx| Ok(ctx.expr(ExprKind::Lambda {
            span: DUMMY, args: ctx.arena.alloc(ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", None),
                ctx.simple_arg("b", None)
            )))?,
            body: ctx.bin_op(ctx.name("a"), Operator::Add, ctx.name("b"))
        })));
        /*
         * Believe it or not, lambdas can actually have default arguments,
         * varargs, and even positional-only specifiers.
         */
        test_expr("lambda a, /, b, *items, c=3: a + c", |ctx| Ok(ctx.expr(ExprKind::Lambda {
            span: DUMMY, args: ctx.arena.alloc(ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", None),
                ctx.arg("b", None, None, ArgumentStyle::PositionalOnly),
                ctx.arg("items", None, None, ArgumentStyle::Vararg),
                ctx.keyword_arg("c", None, Some(ctx.int(3)))
            )))?,
            body: ctx.bin_op(ctx.name("a"), Operator::Add, ctx.name("c"))
        })))
    }
    #[test]
    fn yield_expr() {
        test_expr("1 + (yield a)", |ctx| Ok(ctx.bin_op(
            ctx.int(1), Operator::Add, ctx.expr(ExprKind::Yield {
                span: DUMMY,
                value: Some(ctx.name("a"))
            })
        )));
        test_expr("2 + (yield)", |ctx| Ok(ctx.bin_op(
            ctx.int(2), Operator::Add, ctx.expr(ExprKind::Yield {
                span: DUMMY,
                value: None
            })
        )));
        test_expr("(yield a, b,)", |ctx| Ok(ctx.expr(ExprKind::Yield {
            span: DUMMY,
            value: Some(ctx.expr(ExprKind::Tuple {
                span: DUMMY,
                ctx: ExprContext::Load,
                elts: vec!(ctx, ctx.name("a"), ctx.name("b"))
            }))
        })));
    }
}
