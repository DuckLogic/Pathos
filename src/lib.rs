use alloc::Allocator;
use ast::constants::ConstantPool;
use lexer::{PythonLexer};
use parse::{PythonParser, parser::{Parser, ParseError}};

pub mod alloc;
pub mod lexer;
pub mod ast;
mod parse;
#[cfg(feature = "unicode-named-escapes")]
mod unicode_names;

/// The mode of operation to parse the code in
///
/// Indicates the top level item to parse
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    Expression,
}

pub fn parse<'a, 'src>(
    arena: &'a Allocator,
    text: &'src str,
    mode: ParseMode
) -> Result<ast::tree::Mod<'a>, ParseError> {
    let lexer = PythonLexer::new(arena, text);
    let mut parser = PythonParser::new(
        arena,
        Parser::new(arena, lexer)
    );
    match mode {
        ParseMode::Expression => {
            let res = parser.expression()?;
            Ok(ast::tree::Mod::Expression { body: res })
        }
    }
}
