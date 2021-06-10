use crate::ast::constants::ConstantPool;
use crate::ast::tree::ExprContext;
use crate::ast::Ident;

use crate::lexer::Token;
use self::parser::{ParseError, IParser, Parser};

use crate::alloc::Allocator;

pub use self::expr::ExprPrec;

pub mod parser;
mod expr;


pub struct PythonParser<'src, 'a> {
    pub arena: &'a Allocator,
    pub parser: Parser<'src, 'a>,
    expression_context: ExprContext,
    pool: ConstantPool<'a>
}
impl<'src, 'a> PythonParser<'src, 'a> {
    pub fn new(arena: &'a Allocator, parser: Parser<'src, 'a>) -> Self {
        PythonParser {
            arena, parser,
            expression_context: Default::default(),
            pool: ConstantPool::new(arena)
        }
    }
    #[inline]
    pub fn parse_ident(&mut self) -> Result<&'a Ident<'a>, ParseError> {
        let span = self.parser.current_span();
        Ok(&*self.arena.alloc(Ident::from_raw(span, self.parser.expect_map(&"an identifier", |token| match token.kind {
            Token::Ident(ident) => Some(ident),
            _ => None
        })?)?)?)
    }
}
impl<'src, 'a> IParser<'src, 'a> for PythonParser<'src, 'a> {
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