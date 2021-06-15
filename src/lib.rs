#![feature(
    backtrace, // Better errors ;)\
    variant_count, // Used to convert int -> enum
    const_fn_transmute, const_option, // Used for const ExprPrec::from_int().unwrap()
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
use crate::ast::ident::SymbolTable;

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
    pool: &mut ConstantPool<'a>,
    symbol_table: &mut SymbolTable<'a>
) -> Result<ast::tree::Mod<'a>, ParseError> {
    let lexer = PythonLexer::new(arena, symbol_table, text);
    let mut parser = PythonParser::new(arena, Parser::new(arena, lexer)?, pool, );
    match mode {
        ParseMode::Expression => {
            let res = parser.expression()?;
            Ok(ast::tree::Mod::Expression { body: res })
        }
    }
}
