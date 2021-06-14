//! A in-memory representation of the AST
//!
//! This is automatically generated from the ASDL file

use crate::lexer::Token;
use crate::parse::ExprPrec;

use super::{Ident, Span, Spanned};
use super::constants::Constant;

use educe::Educe;

// Original file was automatically generated by asdl/rust.py.


#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub enum Mod<'a> {
    Module {
        body: &'a [Stmt<'a>],
        type_ignores: &'a [TypeIgnore<'a>],
    },
    Interactive {
        body: &'a [Stmt<'a>],
    },
    Expression {
        body: Expr<'a>,
    },
    FunctionType {
        argtypes: &'a [Expr<'a>],
        returns: Expr<'a>,
    },
}
impl<'a> Mod<'a> {
    #[inline]
    pub fn as_expression(&self) -> Option<Expr<'a>> {
        match *self {
            Mod::Expression { body } => Some(body),
            _ => None
        }
    }
}
/// A reference to a statement
pub type Stmt<'a> = &'a StmtKind<'a>;

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub enum StmtKind<'a> {
    FunctionDef {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        name: Ident<'a>,
        args: &'a Arguments<'a>,
        body: &'a [Stmt<'a>],
        decorator_list: &'a [Expr<'a>],
        returns: Option<Expr<'a>>,
        type_comment: Option<&'a str>,

    },
    AsyncFunctionDef {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        name: Ident<'a>,
        args: &'a Arguments<'a>,
        body: &'a [Stmt<'a>],
        decorator_list: &'a [Expr<'a>],
        returns: Option<Expr<'a>>,
        type_comment: Option<&'a str>,

    },
    ClassDef {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        name: Ident<'a>,
        bases: &'a [Expr<'a>],
        keywords: &'a [Keyword<'a>],
        body: &'a [Stmt<'a>],
        decorator_list: &'a [Expr<'a>],

    },
    Return {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Option<Expr<'a>>,

    },
    Delete {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        targets: &'a [Expr<'a>],
    },
    Assign {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        targets: &'a [Expr<'a>],
        value: Expr<'a>,
        type_comment: Option<&'a str>,
    },
    AugAssign {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        target: Expr<'a>,
        op: Operator,
        value: Expr<'a>,
    },
    AnnAssign {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        target: Expr<'a>,
        annotation: Expr<'a>,
        value: Option<Expr<'a>>,
        simple: bool,
    },
    For {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        target: Expr<'a>,
        iter: Expr<'a>,
        body: &'a [Stmt<'a>],
        orelse: &'a [Stmt<'a>],
        type_comment: Option<&'a str>,
    },
    AsyncFor {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        target: Expr<'a>,
        iter: Expr<'a>,
        body: &'a [Stmt<'a>],
        orelse: &'a [Stmt<'a>],
        type_comment: Option<&'a str>,
    },
    While {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        test: Expr<'a>,
        body: &'a [Stmt<'a>],
        orelse: &'a [Stmt<'a>],
    },
    If {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        test: Expr<'a>,
        body: &'a [Stmt<'a>],
        orelse: &'a [Stmt<'a>],
    },
    With {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        items: &'a [WithItem<'a>],
        body: &'a [Stmt<'a>],
        type_comment: Option<&'a str>,
    },
    AsyncWith {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        items: &'a [WithItem<'a>],
        body: &'a [Stmt<'a>],
        type_comment: Option<&'a str>,

    },
    Match {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        subject: Expr<'a>,
        cases: &'a [MatchCase<'a>],
    },
    Raise {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        exc: Option<Expr<'a>>,
        cause: Option<Expr<'a>>,

    },
    Try {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        body: &'a [Stmt<'a>],
        handlers: &'a [ExceptHandler<'a>],
        orelse: &'a [Stmt<'a>],
        finalbody: &'a [Stmt<'a>],
    },
    Assert {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        test: Expr<'a>,
        msg: Option<Expr<'a>>,

    },
    Import {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        names: &'a [Alias<'a>],
    },
    ImportFrom {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        module: Option<Ident<'a>>,
        names: &'a [Alias<'a>],
        level: Option<i32>,
    },
    Global {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        names: &'a [Ident<'a>],
    },
    Nonlocal {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        names: &'a [Ident<'a>],
    },
    Expr {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
    },
    Pass {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
    },
    Break {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
    },
    Continue {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
    },
}
impl<'a> Spanned for StmtKind<'a> {
    fn span(&self) -> Span {
        match *self {
            StmtKind::FunctionDef { span, .. } |
            StmtKind::AsyncFunctionDef { span, .. } |
            StmtKind::ClassDef { span, .. } |
            StmtKind::Return { span, .. } |
            StmtKind::Delete { span, .. } |
            StmtKind::Assign { span, .. } |
            StmtKind::AugAssign { span, .. } |
            StmtKind::AnnAssign { span, .. } |
            StmtKind::For { span, .. } |
            StmtKind::AsyncFor { span, .. } |
            StmtKind::While { span, .. } |
            StmtKind::If { span, .. } |
            StmtKind::With { span, .. } |
            StmtKind::AsyncWith { span, .. } |
            StmtKind::Match { span, .. } |
            StmtKind::Raise { span, .. } |
            StmtKind::Try { span, .. } |
            StmtKind::Assert { span, .. } |
            StmtKind::Import { span, .. } |
            StmtKind::ImportFrom { span, .. } |
            StmtKind::Global { span, .. } |
            StmtKind::Nonlocal { span, .. } |
            StmtKind::Expr { span, .. } |
            StmtKind::Pass { span, .. } |
            StmtKind::Break { span, .. } |
            StmtKind::Continue { span, .. } => span,
        }
    }
}
/// A reference to an expression
pub type Expr<'a> = &'a ExprKind<'a>;

/// The type of a python expression
#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub enum ExprKind<'a> {
    BoolOp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        op: Boolop,
        values: &'a [Expr<'a>],
    },
    NamedExpr {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        target: Expr<'a>,
        value: Expr<'a>,
    },
    BinOp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        left: Expr<'a>,
        op: Operator,
        right: Expr<'a>,
    },
    UnaryOp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        op: Unaryop,
        operand: Expr<'a>,
    },
    Lambda {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        args: &'a Arguments<'a>,
        body: Expr<'a>,

    },
    IfExp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        test: Expr<'a>,
        body: Expr<'a>,
        orelse: Expr<'a>,

    },
    Dict {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elements: &'a [(Expr<'a>, Expr<'a>)],
    },
    Set {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elts: &'a [Expr<'a>],
    },
    ListComp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elt: Expr<'a>,
        generators: &'a [Comprehension<'a>],
    },
    SetComp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elt: Expr<'a>,
        generators: &'a [Comprehension<'a>],
    },
    DictComp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        key: Expr<'a>,
        value: Expr<'a>,
        generators: &'a [Comprehension<'a>],
    },
    GeneratorExp {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elt: Expr<'a>,
        generators: &'a [Comprehension<'a>],
    },
    Await {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
    },
    Yield {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Option<Expr<'a>>,

    },
    YieldFrom {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,

    },
    Compare {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        left: Expr<'a>,
        ops: &'a [Cmpop],
        comparators: &'a [Expr<'a>],
    },
    Call {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        func: Expr<'a>,
        args: &'a [Expr<'a>],
        keywords: &'a [Keyword<'a>],
    },
    FormattedValue {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
        conversion: Option<char>,
        format_spec: Option<Expr<'a>>,
    },
    JoinedStr {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        values: &'a [Expr<'a>],
    },
    Constant {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Constant<'a>,
        kind: Option<&'a str>,
    },
    Attribute {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
        attr: Ident<'a>,
        ctx: ExprContext,
    },
    Subscript {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
        slice: Expr<'a>,
        ctx: ExprContext,
    },
    Starred {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
        ctx: ExprContext,
    },
    Name {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        id: Ident<'a>,
        ctx: ExprContext,
    },
    List {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elts: &'a [Expr<'a>],
        ctx: ExprContext,
    },
    Tuple {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        elts: &'a [Expr<'a>],
        ctx: ExprContext,
    },
    Slice {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        lower: Option<Expr<'a>>,
        upper: Option<Expr<'a>>,
        step: Option<Expr<'a>>,
    },
}
impl<'a> Spanned for ExprKind<'a> {
    fn span(&self) -> Span {
        match *self {
            ExprKind::BoolOp { span, .. } |
            ExprKind::NamedExpr { span, .. } |
            ExprKind::BinOp { span, .. } |
            ExprKind::UnaryOp { span, .. } |
            ExprKind::Lambda { span, .. } |
            ExprKind::IfExp { span, .. } |
            ExprKind::Dict { span, .. } |
            ExprKind::Set { span, .. } |
            ExprKind::ListComp { span, .. } |
            ExprKind::SetComp { span, .. } |
            ExprKind::DictComp { span, .. } |
            ExprKind::GeneratorExp { span, .. } |
            ExprKind::Await { span, .. } |
            ExprKind::Yield { span, .. } |
            ExprKind::YieldFrom { span, .. } |
            ExprKind::Compare { span, .. } |
            ExprKind::Call { span, .. } |
            ExprKind::FormattedValue { span, .. } |
            ExprKind::JoinedStr { span, .. } |
            ExprKind::Constant { span, .. } |
            ExprKind::Attribute { span, .. } |
            ExprKind::Subscript { span, .. } |
            ExprKind::Starred { span, .. } |
            ExprKind::Name { span, .. } |
            ExprKind::List { span, .. } |
            ExprKind::Tuple { span, .. } |
            ExprKind::Slice { span, .. } => span,
        }
    }
}
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ExprContext {
    Load=1,
    Store=2,
    Del=3,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Boolop {
    And=1,
    Or=2,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Operator {
    Add,
    Sub,
    Mult,
    MatMult,
    Div,
    Mod,
    Pow,
    LShift,
    RShift,
    BitOr,
    BitXor,
    BitAnd,
    FloorDiv,
}
impl Operator {
    #[inline]
    pub fn is_right_associative(&self) -> bool {
        matches!(*self, Operator::Pow)
    }
    #[inline]
    pub fn from_token(tk: &Token) -> Option<Self> {
        Some(match *tk {
            Token::Plus => Operator::Add,
            Token::Minus => Operator::Sub,
            Token::Star => Operator::Mult,
            Token::At => Operator::MatMult,
            Token::Slash => Operator::Div,
            Token::DoubleSlash => Operator::FloorDiv,
            Token::DoubleStar => Operator::Pow,
            Token::LeftShift => Operator::LShift,
            Token::RightShift => Operator::RShift,
            Token::BitwiseOr => Operator::BitOr,
            Token::BitwiseXor => Operator::BitXor,
            Token::Ampersand => Operator::BitAnd,
            _ => return None
        })
    }
    #[inline]
    pub fn precedence(self) -> crate::parse::ExprPrec {
        match self {
            Operator::Add | Operator::Sub => ExprPrec::Term,
            Operator::Mult | Operator::MatMult |
            Operator::Div | Operator::Mod |
            Operator::FloorDiv => ExprPrec::Factor,
            Operator::Pow => ExprPrec::Exponentation,
            Operator::LShift | Operator::RShift => ExprPrec::Shifts,
            Operator::BitOr => ExprPrec::BitwiseOr,
            Operator::BitXor => ExprPrec::BitwiseXor,
            Operator::BitAnd => ExprPrec::BitwiseAnd
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Unaryop {
    Invert,
    Not,
    UAdd,
    USub,
}
impl Unaryop {
    #[inline]
    pub fn from_token(tk: &Token) -> Option<Self> {
        Some(match *tk {
            Token::BitwiseInvert => Unaryop::Invert,
            Token::Not => Unaryop::Not,
            Token::Plus => Unaryop::UAdd,
            Token::Minus => Unaryop::USub,
            _ => return None
        })
    }
    #[inline]
    pub fn precedence(self) -> ExprPrec {
        match self {
            Unaryop::Invert |
            Unaryop::Not |
            Unaryop::UAdd |
            Unaryop::USub => ExprPrec::Unary,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Cmpop {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Is,
    IsNot,
    In,
    NotIn,
}
impl Cmpop {
    #[inline]
    pub fn precedence(self) -> ExprPrec {
        match self {
            Cmpop::Eq |
            Cmpop::NotEq |
            Cmpop::Lt |
            Cmpop::LtE |
            Cmpop::Gt |
            Cmpop::GtE |
            Cmpop::Is |
            Cmpop::IsNot |
            Cmpop::In |
            Cmpop::NotIn => ExprPrec::Comparisons
        }
    }
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct Comprehension<'a> {
    pub target: Expr<'a>,
    pub iter: Expr<'a>,
    pub ifs: &'a [Expr<'a>],
    pub is_async: bool,
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct ExceptHandler<'a> {
    /// The span of the source
    #[educe(Hash(ignore))]
    #[educe(PartialEq(ignore))]
    span: Span,
    ast_type: Option<Expr<'a>>,
    name: Option<Ident<'a>>,
    body: &'a [Stmt<'a>],
}
impl<'a> Spanned for ExceptHandler<'a> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}
#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct Arguments<'a> {
    pub posonlyargs: &'a [Arg<'a>],
    pub args: &'a [Arg<'a>],
    pub vararg: Option<&'a Arg<'a>>,
    pub kwonlyargs: &'a [Arg<'a>],
    pub kw_defaults: &'a [Expr<'a>],
    pub kwarg: Option<&'a Arg<'a>>,
    pub defaults: &'a [Expr<'a>],
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct Arg<'a> {
    pub arg: Ident<'a>,
    pub annotation: Option<Expr<'a>>,
    pub type_comment: Option<&'a str>,
    #[educe(Hash(ignore))]
    #[educe(PartialEq(ignore))]
    pub span: Span,
}
impl<'a> Spanned for Arg<'a> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct Keyword<'a> {
    pub arg: Option<Ident<'a>>,
    pub value: Expr<'a>,
    #[educe(Hash(ignore))]
    #[educe(PartialEq(ignore))]
    pub span: Span,
}
impl<'a> Spanned for Keyword<'a> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct Alias<'a> {
    pub name: Ident<'a>,
    pub asname: Option<Ident<'a>>,
    #[educe(Hash(ignore))]
    #[educe(PartialEq(ignore))]
    pub span: Span,
}
impl<'a> Spanned for Alias<'a> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct WithItem<'a> {
    pub context_expr: Expr<'a>,
    pub optional_vars: Option<Expr<'a>>,
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub struct MatchCase<'a> {
    pub pattern: &'a Pattern<'a>,
    pub guard: Option<Expr<'a>>,
    pub body: &'a [Stmt<'a>],
}

#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq, Hash)]
pub enum Pattern<'a> {
    MatchValue {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Expr<'a>,
    },
    MatchSingleton {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        value: Constant<'a>,
    },
    MatchSequence {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        patterns: &'a [Pattern<'a>],
    },
    MatchMapping {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        keys: &'a [Expr<'a>],
        patterns: &'a [Pattern<'a>],
        rest: Option<Ident<'a>>,
    },
    MatchClass {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        cls: Expr<'a>,
        patterns: &'a [Pattern<'a>],
        kwd_attrs: &'a [Ident<'a>],
        kwd_patterns: &'a [Pattern<'a>],
    },
    MatchStar {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        name: Option<Ident<'a>>,
    },
    MatchAs {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        pattern: Option<&'a Pattern<'a>>,
        name: Option<Ident<'a>>,
    },
    MatchOr {
        /// The span of the source
        #[educe(Hash(ignore))]
        #[educe(PartialEq(ignore))]
        span: Span,
        patterns: &'a [Pattern<'a>],
    },
}
impl<'a> Spanned for Pattern<'a> {
    fn span(&self) -> Span {
        match *self {
            Pattern::MatchValue { span, .. } |
            Pattern::MatchSingleton { span, .. } |
            Pattern::MatchSequence { span, .. } |
            Pattern::MatchMapping { span, .. } |
            Pattern::MatchClass { span, .. } |
            Pattern::MatchStar { span, .. } |
            Pattern::MatchAs { span, .. } |
            Pattern::MatchOr { span, .. } => span,
        }
    }
}
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct TypeIgnore<'a> {
    pub lineno: i32,
    pub tag: &'a str,
}

