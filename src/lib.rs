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
#[allow(clippy::all)]
mod unicode_names;

pub use parse::errors::ParseError;
use crate::ast::ident::SymbolTable;

/// The mode of operation to parse the code in
///
/// Indicates the top level item to parse
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    Expression,
    Module
}

pub fn parse_text<'a, 'src>(
    arena: &'a Allocator,
    text: &'src str,
    mode: ParseMode,
    pool: &mut ConstantPool<'a>,
    symbol_table: &mut SymbolTable<'a>
) -> Result<ast::tree::PythonAst<'a>, ParseError> {
    let lexer = PythonLexer::new(
        arena,
        std::mem::replace(symbol_table, SymbolTable::new(arena)), // take it from them
        text
    );
    let mut parser = PythonParser::new(arena, Parser::new(arena, lexer)?, pool);
    let res= parser.parse_top_level(mode).map_err(|err| {
        err.with_line_numbers(parser.parser.line_number_tracker())
    })?;
    parser.parser.expect_end_of_input()?;
    // give back the symbol table we took
    *symbol_table = parser.parser.into_lexer().into_symbol_table();
    Ok(res)
}
