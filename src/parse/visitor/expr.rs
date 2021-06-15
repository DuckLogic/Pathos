//! Visitors for expressions

use crate::ast::tree::{Operator, Expr, ExprKind, ExprContext, UnaryOp, Comprehension};
use crate::ast::{Span, AstBuilder, Ident, Constant};
use crate::alloc::AllocError;
use crate::parse::errors::ParseVisitError;
use crate::parse::visitor::target::TargetVisitor;

/// Visits an expression
///
/// An ExpressionVisitor can also be viewed as a more general form of a [TargetVisitor],
/// that supports visiting any type of expression (and not just assignment targets).
pub trait ExprVisitor<'a>: TargetVisitor<Target=Self::Expr> {
    /// The result of successfully visiting an expression
    type Expr: Sized;
    /// The error type
    type Err: Sized;

    /// Visit a binary operator
    fn visit_bin_op(&mut self, span: Span, left: Self::Expr, op: Operator, right: Self::Expr) -> Result<Self::Expr, Self::Err>;

    /// Visit a unary operator
    fn visit_unary_op(&mut self, span: Span, op: UnaryOp, operand: Expr<'a>) -> Result<Self::Expr, Self::Err>;

    /// Visit a constant expression
    ///
    /// NOTE: Kind is "u" for unicode strings, and `None` otherwise
    fn visit_constant(&mut self, span: Span, kind: Option<&'a str>, constant: Constant<'a>) -> Result<Self::Expr, Self::Err>;
}
impl<'a> ExprVisitor<'a> for AstBuilder<'a> {
    type Expr = Expr<'a>;
    type Err = AllocError;

    #[inline]
    fn visit_bin_op(&mut self, span: Span, left: Self::Expr, op: Operator, right: Self::Expr) -> Result<Self::Expr, Self::Err> {
        self.expr(ExprKind::BinOp {
            left, op, right, span
        })
    }

    #[inline]
    fn visit_unary_op(&mut self, span: Span, op: UnaryOp, operand: Expr<'a>) -> Result<Self::Expr, Self::Err> {
        self.expr(ExprKind::UnaryOp { span, op, operand })
    }

    #[inline]
    fn visit_constant(&mut self, span: Span, kind: Option<&'a str>, constant: Constant<'a>) -> Result<Self::Expr, Self::Err> {
        self.expr(ExprKind::Constant {
            span, kind, value: constant
        })
    }
}

/// A list of comprehensions.
///
/// There always must be at least one comprehension,
/// called the primary comprehension.
/// Anything else is "secondary".
///
/// Python comprehensions allow nesting ([e1 * e2 for e1 in l1 for e2 in l2]),
/// in which case python will visit all possible combinations of l1 and l2.
pub trait ComprehensionListVisitor<'a> {
    type Err: ParseVisitError<'a>;
    type CompList: Sized;

    type PrimaryComp;
    type PrimaryVisit: ComprehensionVisitor<'a, Err=Self::Err, Comp=Self::PrimaryComp>;
    /// Visit the primary comprehension
    fn visit_primary_comprehension(&mut self) -> Self::PrimaryVisit;
    fn finish_primary_comprehension(&mut self, comp: Self::PrimaryVisit::Comp) -> Result<(), Self::Err>;

    type SecondaryComp;
    type SecondaryVisit: ComprehensionVisitor<'a, Err=Self::Err, Comp=Self::SecondayComp>;
    fn visit_secondary_comprehension(&mut self) -> Self::SecondaryVisit;
    fn finish_secondary_comprehension(&mut self, comp: Self::SecondaryVisit::Comp) -> Result<(), Self::Err>;

    fn finish(&mut self, primary: Self::PrimaryVisit::Result);
}
/// Visits a collection comprehension
///
/// NOTE: This only visits a *single* comprehension.
/// If there may be multiple, see [ComprehensionList]
pub trait ComprehensionVisitor<'a> {
    type Err: ParseVisitError;
    /// The type of the resulting comprehension
    type Comp: Sized;

    type TargetVisit: TargetVisitor<'a>;
    /// Visit the target that the iterated variable is bound to
    fn visit_target(&mut self) -> Self::TargetVisit;

    type IterVisit: ExprVisitor<'a>;
    /// Visit the expression being iterated over
    fn visit_iter(&mut self) -> Self::IterVisit;

    type CondVisit: ExprVisitor<'a>;
    /// Visit the conditional used to filter elements.
    ///
    /// There can potentially be multiple conditionals corresponding to a single comprehension.
    ///
    /// This method is optional, and may not be called
    fn visit_condition(&mut self) -> Self::CondVisit;

    /// Mark the comprehension as an async comprehension
    fn mark_async(&mut self) -> Result<(), Self::Err>;

    /// Finish parsing the comprehension
    fn finish(self, span: Span) -> Result<Self::Comp, Self::Err>;

    /// The type of a secondary comprehension
    type SecondaryComp: Sized;
    /// The type of secondary visitor.
    ///
    /// This can potentially be a different type.
    ///
    /// However, sub-visitors can never have their own sub-visitors.
    /// Multiple sub-comprehensions are always called in a loop, never recursively.
    type SecondaryVisit: ComprehensionVisitor<'a, Err=Self::Err, Comp=Self::SecondaryComp, SecondaryVisit=!>;
    /// Visit a secondary comprehension
    ///
    /// This is used to implement product comprehensions
    /// like `[e1 * e2 for e1 in l1 for e2 in l2]`
    /// In this case, the primary visitor would get "for e1 in l1"
    /// and the "secondary visitor" would get "for e2 in l2"
    fn visit_secondary_comprehension(&mut self) -> Self::SecondaryVisit;
    /// Finish visiting a secondary comprehension
    fn finish_secondary_comprehension(&mut self, res: Self::SecondaryVisit::Comp) -> Result<(), Self::Err>;
}
pub struct AstComprehensionBuilder<'a, const SINGLE: bool> {
    builder: AstBuilder<'a>,
    iter: Option<Expr<'a>>,
    target: Option<Expr<'a>>,
    ifs: crate::alloc::Vec<'a, Expr<'a>>,
    is_async: bool,
    // The sub-comprehensions to visit
    sub_comprehensions: Option<crate::alloc::Vec<'a, Comprehension<'a>>>
}
impl<'a> AstComprehensionBuilder<'a> {
    #[inline]
    pub fn new(builder: AstBuilder<'a>) -> AstComprehensionBuilder {
        let sub_comprehensions = crate::alloc::Vec::new(builder.arena);
        let ifs = crate::alloc::Vec::new(builder.arena);
        AstComprehensionBuilder {
            builder, iter: None, target: None,
            ifs, is_async: false,
            sub_comprehensions: Some(sub_comprehensions)
        }
    }
}
impl<'a> ComprehensionVisitor<'a> for AstComprehensionBuilder<'a> {
    type Err = AllocError;
    type Comp = Comprehension<'a>;
    type TargetVisit = AstBuilder<'a>;

    #[inline]
    fn visit_target(&mut self) -> Self::TargetVisit {
        self.builder.clone()
    }

    type IterVisit = AstBuilder<'a>;

    #[inline]
    fn visit_iter(&mut self) -> Self::IterVisit {
        self.builder.clone()
    }

    type CondVisit = AstBuilder<'a>;

    #[inline]
    fn visit_condition(&mut self) -> Self::CondVisit {
        self.builder.clone()
    }

    #[inline]
    fn mark_async(&mut self) -> Result<(), Self::Err> {
        self.is_async = true;
        Ok(())
    }

    fn finish(self, _span: Span) -> Result<Self::Comp, Self::Err> {
        Ok(Comprehension {
            target: self.target.unwrap(),
            iter: self.iter.unwrap(),
            ifs: &*self.ifs.into_slice(),
            is_async: self.is_async
        })
    }

    type SecondaryComp = Comprehension<'a>;
    type SecondaryVisit = AstComprehensionBuilder<'a>;

    fn visit_secondary_comprehension(&mut self) -> Self::SecondaryVisit {
        let sub_comprehensions = self.sub_comprehensions.take().unwrap();
        let mut sub_builder = AstComprehensionBuilder::new(self.builder);
    }

    fn finish_secondary_comprehension(&mut self, res: <<_ as ComprehensionVisitor>::SecondaryVisit as ComprehensionVisitor<'a>>::Comp) -> Result<(), Self::Err> {
        todo!()
    }
}
