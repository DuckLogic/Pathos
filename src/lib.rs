#![feature(
    backtrace, // Better errors ;)\
    variant_count, // Used to convert int -> enum
    const_option, // Used for const ExprPrec::from_int().unwrap()
)]
#![feature(cell_update)]

use alloc::Allocator;
use crate::lexer::Lexer;

pub mod alloc;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod errors;

/// The base type of all supported parsers
pub trait Pathos<'a> {
    type Lexer: Lexer<'a>;
    fn alloc(&self) -> &'a Allocator;
}
