use crate::ast::AstVisitor;
use crate::ast::tree::ExprContext;

pub mod parser;
mod expr;


pub struct PythonParser<'src, 'a, 'v, V: AstVisitor<'a>> {
    pub parser: self::parser::Parser<'src, 'a>,
    pub visitor: &'v mut V,
    expression_context: ExprContext
}
impl Default for ExprContext {
    #[inline]
    fn default() -> Self {
        ExprContext::Load
    }
}