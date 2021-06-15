use crate::parse::errors::ParseVisitError;
use crate::ast::{Ident, AstBuilder, Span};
use crate::ast::tree::{ExprKind, Expr, ExprContext};
use crate::alloc::AllocError;
use crate::parse::ExprPrec;

/// Visits a "target" of an assignment
///
/// Targets can also be viewed as a more restricted form of expressions.
/// In fact, that is how they are implemented in the CPython AST.
///
/// See the section in the official reference on "assignments":
/// <https://docs.python.org/3.10/reference/simple_stmts.html#assignment-statements>
pub trait TargetVisitor<'a> {
    /// The specific error type
    type Err: ParseVisitError;
    /// The result of successfully visiting a target
    type Target: Sized;

    /// Visit a named target
    ///
    /// This is the most common type of target
    fn visit_name(&mut self, name: Ident<'a>) -> Result<Self::Target, Self::Err>;

    type TupleVisit;
    type ListVisit;

    /// Visits a tuple target
    fn visit_tuple(&mut self) -> Self::TupleVisit;
    /// Visits a list target
    fn visit_list(&mut self) -> Self::ListVisit;
}
/// Visits a list of targets,
/// in other words, a combination of [TargetVisitor]s
pub trait TargetListVisitor<'a> {
    type Err: ParseVisitError;
    /// The type of intermediate results
    type Intermediate: Sized;
    /// The type of successful result
    type ResultList: Sized;
    /// Visit a named target
    ///
    /// This is special-cased because it is so common.
    /// It can avoid some implementation complexity
    #[inline]
    fn visit_named_target(&mut self, name: Ident<'a>) -> Result<(), Self::Err> {
        self.finished_visiting_target(self.target_visitor().visit_name(name)?)
    }
    type TargetVisit: TargetVisitor<'a, Err=Self::Err, Target=Self::Intermediate>;
    /// Visit a target in the list
    ///
    /// This is the general case
    fn target_visitor(&mut self) -> Self::TargetVisit;
    /// Called when done visiting a target, taking the immediate result of the target
    fn finished_visiting_target(&mut self, res: Self::Intermediate) -> Result<(), Self::Err>;
    /// Called when completely done visiting the entire collection
    fn finish(self, span: Span) -> Result<Self::ResultList, Self::Err>;
}
enum CollectionKind {
    Tuple,
    Set,
    List,
}
/// A [TargetListVisitor], that builds a specified [CollectionKind]
///
/// Cannot build a dictionary. That needs a special [AstDictBuilder]
pub struct AstCollectionBuilder<'a> {
    pub builder: AstBuilder<'a>,
    pub kind: CollectionKind,
    pub res: crate::alloc::Vec<'a, Expr<'a>>
}
impl<'a> TargetListVisitor<'a> for AstCollectionBuilder {
    type Err = AllocError;
    type Intermediate = Expr<'a>;
    type ResultList = Expr<'a>;

    type TargetVisit = AstBuilder<'a>;

    #[inline]
    fn target_visitor(&mut self) -> Self::TargetVisit {
        self.builder.clone()
    }

    fn finished_visiting_target(&mut self, expr: Self::Intermediate) -> Result<(), Self::Err> {
        self.res.push(expr)?;
        Ok(())
    }

    #[inline]
    fn finish(&mut self, span: Span) -> Result<Self::ResultList, Self::Err> {
        let elts = &*self.res.into_slice();
        let ctx = self.ctx;
        match self.kind {
            CollectionKind::Tuple => {
                self.builder.expr(ExprKind::Tuple { span, elts, ctx })
            },
            CollectionKind::Set => {
                self.builder.expr(ExprKind::Set { elts, span })
            }
            CollectionKind::List => {
                self.builder.expr(ExprKind::List { elts, span, ctx })
            }
        }
    }

}
impl<'a> TargetVisitor<'a> for AstBuilder<'a> {
    type Err = AllocError;
    type Target = Expr<'a>;
    #[inline]
    fn visit_name(&mut self, ident: Ident<'a>) -> Result<Self::Expr, Self::Err> {
        self.expr(ExprKind::Name {
            span: ident.span, id: ident, ctx
        })
    }

    type TupleVisit = AstCollectionBuilder<'a>;
    type ListVisit = AstCollectionBuilder<'a>;

    #[inline]
    fn visit_tuple(&mut self) -> Self::TupleVisit {
        AstCollectionBuilder { builder, kind: CollectionKind::Tuple, res: crate::alloc::Vec::new(self.arena) }
    }

    #[inline]
    fn visit_list(&mut self) -> Self::ListVisit {
        AstCollectionBuilder { builder, kind: CollectionKind::List, res: crate::alloc::Vec::new(self.arena) }
    }
}