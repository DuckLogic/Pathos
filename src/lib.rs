#![feature(
    backtrace, // Better errors ;)
)]
use alloc::Allocator;
use ast::constants::ConstantPool;
use lexer::{PythonLexer};
use parse::{PythonParser, parser::{Parser}};

pub mod alloc;
pub mod lexer;
pub mod ast;
mod parse;
#[cfg(feature = "unicode-named-escapes")]
mod unicode_names;

pub use self::parse::parser::ParseError;

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
    mode: ParseMode,
    pool: &mut ConstantPool<'a>
) -> Result<ast::tree::Mod<'a>, ParseError> {
    let lexer = PythonLexer::new(arena, text);
    let mut parser = PythonParser::new(
        arena,
        Parser::new(arena, lexer)?,
        pool
    );
    match mode {
        ParseMode::Expression => {
            let res = parser.expression()?;
            Ok(ast::tree::Mod::Expression { body: res })
        }
    }
}
