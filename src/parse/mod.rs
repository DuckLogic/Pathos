use crate::ast::constants::ConstantPool;
use crate::ast::tree::ExprContext;
use crate::ast::Span;
use crate::ast::ident::{Ident, Symbol, SymbolTable};

use crate::lexer::Token;
use self::parser::{ParseError, IParser, Parser};

use crate::alloc::Allocator;

pub use self::expr::ExprPrec;

pub mod parser;
mod expr;


#[derive(Debug)]
pub struct PythonParser<'src, 'a, 'p> {
    pub arena: &'a Allocator,
    pub parser: Parser<'src, 'a, 'p>,
    expression_context: ExprContext,
    pub pool: &'p mut ConstantPool<'a>,
}
impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    pub fn new(arena: &'a Allocator, parser: Parser<'src, 'a, 'p>, pool: &'p mut ConstantPool<'a>) -> Self {
        PythonParser {
            arena, parser, pool,
            expression_context: Default::default(),
        }
    }
    #[inline]
    pub fn parse_ident(&mut self) -> Result<&'a Ident<'a>, ParseError> {
        let span = self.parser.current_span();
        let raw = self.parser.expect_map(&"an identifier", |token| match token.kind {
            Token::Ident(ident) => Some(ident),
            _ => None
        })?;
        self.convert_ident(span, raw)
    }
}
impl<'src, 'a, 'p> IParser<'src, 'a, 'p> for PythonParser<'src, 'a, 'p> {
    #[inline]
    fn as_mut_parser(&mut self) -> &mut Parser<'src, 'a, 'p> {
        &mut self.parser
    }
    #[inline]
    fn as_parser(&self) -> &Parser<'src, 'a, 'p> {
        &self.parser
    }
}
impl Default for ExprContext {
    #[inline]
    fn default() -> Self {
        ExprContext::Load
    }
}