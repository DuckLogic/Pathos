use errors::ParseError;

use crate::alloc::Allocator;
use crate::ast::constants::ConstantPool;
use crate::ast::ident::Ident;
use crate::ast::tree::ExprContext;
use crate::lexer::Token;

pub use self::expr::ExprPrec;
use self::parser::{IParser, Parser};

pub mod errors;
pub mod parser;
mod expr;


#[derive(Debug)]
pub struct PythonParser<'src, 'a, 'p> {
    pub arena: &'a Allocator,
    pub parser: Parser<'src, 'a>,
    expression_context: ExprContext,
    pub pool: &'p mut ConstantPool<'a>,
}
impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    pub fn new(arena: &'a Allocator, parser: Parser<'src, 'a>, pool: &'p mut ConstantPool<'a>) -> Self {
        PythonParser {
            arena, parser, pool,
            expression_context: Default::default(),
        }
    }
    #[inline]
    pub fn parse_ident(&mut self) -> Result<Ident<'a>, ParseError> {
        let span = self.parser.current_span();
        let symbol = self.parser.expect_map(&"an identifier", |token| match token.kind {
            Token::Ident(ident) => Some(ident),
            _ => None
        })?;
        Ok(Ident {
            symbol, span
        })
    }
}
impl<'src, 'a, 'p> IParser<'src, 'a> for PythonParser<'src, 'a, 'p> {
    #[inline]
    fn as_mut_parser(&mut self) -> &mut Parser<'src, 'a> {
        &mut self.parser
    }
    #[inline]
    fn as_parser(&self) -> &Parser<'src, 'a> {
        &self.parser
    }
}
impl Default for ExprContext {
    #[inline]
    fn default() -> Self {
        ExprContext::Load
    }
}