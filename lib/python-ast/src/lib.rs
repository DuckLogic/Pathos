#![feature(variant_count)]

use std::fmt::{Debug};

pub mod constants;
pub mod tree;
mod prec;

pub use self::prec::ExprPrec;
pub use pathos::alloc::Allocator;
use crate::tree::{ExprKind, Expr};
use pathos::alloc::AllocError;

/// A [ParseVisitor](crate::parse::visitor::ParseVisitor) that constructs
/// an abstract syntax tree
#[derive(Copy, Clone, Debug)]
pub struct AstBuilder<'a> {
    pub arena: &'a Allocator,
}
impl<'a> AstBuilder<'a> {
    /// Create an expression of the specified [ExprKind]
    #[inline]
    pub fn expr(&self, kind: ExprKind<'a>) -> Result<Expr<'a>, AllocError> {
        Ok(self.arena.alloc(kind)?)
    }
}
