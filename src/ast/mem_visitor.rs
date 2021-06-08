use std::option;

/// A marker struct for generating ASTs in-memory
use super::visitor::*;
use super::tree::{self, Stmt, Expr};
use super::{Span, Ident};

#[derive(Copy, Clone, Debug)]
pub enum NeverFails {}

pub struct MemoryVisitor;

pub struct MemoryComprehensionBodyVisitor {
    target: Box<Expr>,
    iter: Box<Expr>,
    ifs: Vec<Stmt>,
    is_async: bool
}
impl<'a> ComprehensionBodyVisitor<'a> for MemoryComprehensionBodyVisitor {
    fn visit_if_stmt(&mut self, stmt: Self::Stmt) -> Result<(), Self::Err> {
        self.ifs.push(stmt);
        Ok(())
    }
    fn finish(self, span: Span) -> Result<Self::Comprehension, Self::Err> {
        Ok(tree::Comprehension {
            target: self.target,
            iter: self.iter,
            ifs: self.ifs,
            is_async: if self.is_async { 1 } else { 0 }
        })
    }

}
pub struct MemoryMatchCaseBodyVisitor {
    pattern: Box<tree::Pattern>,
    guard: Option<Box<tree::Expr>>,
    body: Vec<Stmt>
}
impl<'a> MatchCaseBodyVisitor<'a> for MemoryMatchCaseBodyVisitor {
    fn visit_body_stmt(&mut self, stmt: Self::Stmt) -> Result<(), Self::Err> {
        self.body.push(stmt);
        Ok(())
    }
    fn finish(self, span: Span) -> Result<Self::MatchCase, Self::Err> {
        Ok(tree::MatchCase {
            pattern: self.pattern,
            guard: self.guard, body: self.body
        })
    }
}
impl<'a> AstVisitor<'a> for MemoryVisitor {
    type Ident = Ident;
    type Err = NeverFails;

    #[inline]
    fn create_ident(&mut self, span: Span, ident: &RawIdent<'a>) -> Result<Self::Ident, Self::Err> {
        Ok(Ident::new(span, ident.text()))
    }

    #[inline]
    fn create_keyword(
            &mut self,
            span: Span,
            arg: Option<Self::Ident>,
            value: Self::Expr,
    ) -> Result<tree::Keyword, Self::Err> {
        Ok(tree::Keyword { span, arg: arg, value })
    }

    type ComprehensionBodyVisitor = MemoryComprehensionBodyVisitor;
    fn visit_comprehension(
            &mut self,
            target: Self::Expr,
            iter: Self::Expr,
            is_async: bool,
    ) -> Result<Self::ComprehensionBodyVisitor, Self::Err> {
        Ok(MemoryComprehensionBodyVisitor {
            target, iter, is_async, ifs: Vec::new()
        })
    }
    fn visit_alias(
            &mut self,
            span: Span,
            name: Self::Ident,
            asname: Option<Self::Ident>,
    ) -> Result<Self::Alias, Self::Err> {
        Ok(tree::Alias {
            span, name, asname
        })
    }
    fn visit_withitem(
            &mut self,
            context_expr: Self::Expr,
            optional_vars: Option<Self::Expr>,
    ) -> Result<Self::Withitem, Self::Err> {
        Ok(tree::Withitem {
            context_expr: Box::new(context_expr),
            optional_vars: optional_vars.map(Box::new)
        })
    }
    type MatchCaseBodyVisitor = MemoryMatchCaseBodyVisitor;
    fn visit_match_case(
            &mut self,
            pattern: Self::Pattern,
            guard: Option<Self::Expr>,
    ) -> Result<Self::MatchCaseBodyVisitor, Self::Err> {
        Ok(MemoryMatchCaseBodyVisitor {
            pattern: Box::new(pattern),
            guard: guard.map(Box::new),
            body: Vec::new()
        })
    }

    fn visit_type_ignore(
            &mut self,
            lineno: i32,
            tag: &'a str,
    ) -> Self::TypeIgnore {
        Ok(tree::TypeIgnore::TypeIgnore {
            lineno, tag: tag.into()
        })
    }

    type Mod = tree::Mod;
    type Stmt = tree::Stmt;
    type Expr = tree::Expr;
    type Comprehension = tree::Comprehension;
    type Arguments = tree::Arguments;
    type Arg = tree::Arg;
    type Keyword = tree::Keyword;
    type Alias = tree::Alias;
    type Withitem = tree::Withitem;
    type MatchCase = tree::MatchCase;
    type Pattern = tree::Pattern;
    type TypeIgnore = tree::TypeIgnore;
}
