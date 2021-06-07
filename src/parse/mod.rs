use crate::ast::AstVisitor;
use crate::ast::tree::ExprContext;

use crate::lexer::Token;
use self::parser::{ParseError, Parser};

pub mod parser;
mod expr;


pub struct PythonParser<'p, 'src, 'a, 'v, V: AstVisitor<'a>> {
    pub parser: &'p mut Parser<'src, 'a>,
    pub visitor: &'v mut V,
    expression_context: ExprContext
}
impl<'p, 'src, 'a, 'v, V: AstVisitor<'a>> PythonParser<'p, 'src, 'a, 'v, V> {
    #[inline]
    fn parse_terminated<T, F>(
        &mut self,
        sep: Token<'a>,
        ending: Token<'a>,
        parse_func: F
    ) -> impl Iterator<Item=Result<T, ParseError>>
        where for<'p2, 'v2> F: FnMut(&mut PythonParser<'p2, 'src, 'a, 'v2, V>) -> Result<T, ParseError> {
        let expression_context = self.expression_context;
        self.parser.parse_terminated(sep, ending, |parser| {
            let mut parser = PythonParser { parser, visitor, expression_context };
            parse_func(&mut parser)
        })
    }
}
impl Default for ExprContext {
    #[inline]
    fn default() -> Self {
        ExprContext::Load
    }
}