//! Generic parse tree visitors
//!
//! This can be used with an [AstBuilder](crate::ast::builder::AstBuilder) to build
//! an in-memory AST, or it can be used (separately) to generate some sort of bytecode
//! without constructing an intermediate tree.

use std::fmt::Debug;

pub use crate::parse::visitor::expr::ExprVisitor;
use crate::alloc::AllocError;
use crate::ast::AstBuilder;
use crate::parse::errors::ParseVisitError;

mod expr;
mod target;

/// Visits the result of parsing some python file
pub trait ParseVisitor<'a>: Debug {
    /// The type of errors returned by this parser
    type Err: ParseVisitError;
    /// The sub-type for visiting expressions
    type ExprVisit: ExprVisitor<Err=Self::Err> = AstBuilder<'a>;
    /// Begin to visit an expression
    fn expr_visitor(&mut self) -> Self::ExprVisit;
}
impl<'a> ParseVisitor<'a> for AstBuilder<'a> {
    type Err = AllocError;

    #[inline]
    fn expr_visitor(&mut self) -> ExprVisit {
        self.clone()
    }
}