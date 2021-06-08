use crate::ast::tree::ExprContext;

use crate::lexer::Token;
use self::parser::{ParseError, IParser, Parser};

pub mod parser;
mod expr;


pub struct PythonParser<'src, 'a> {
    pub parser: Parser<'src, 'a>,
    expression_context: ExprContext
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