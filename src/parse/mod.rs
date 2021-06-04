use crate::lexer::{PythonLexer, Token};
use crate::ast::Ident;

mod parser;

impl S

fn ident(lex: &mut Parser<'a>) -> Result<&'a Ident<'a>, Error<'a>> {
    match lex.next() {
        Token::Ident(inner) => Ok(inner),
        _ => Err()
    }
}