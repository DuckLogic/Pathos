//! A set of macros to build an AST
#[macro_export]
macro_rules! ident {
    ($ctx:expr, $span:expr, $raw:expr) => ($ctx.arena.alloc(Ident::from_raw($span, $raw)?)?);
    ($ctx:expr, $span:expr, $raw:literal) => ($ctx.arena.alloc(Ident::from_raw($span, RawIdent::from_static_str($raw))?)?)
}
/// Create an [Expr](crate::ast::tree::Expr)
#[macro_export]
macro_rules! expr {
    ($ctx:expr, Expr :: $variant:ident { span: $span:expr, $($field_name:ident : $field_value:expr),+ }) => (
        $ctx.arena.alloc(ExprKind::$variant { span: $span, $($field_name: $field_value),* })?
    );
}
/// Create a [Stmt](crate::ast::tree::Stmt)
#[macro_export]
macro_rules! stmt {
    ($ctx:expr, Stmt :: $variant:ident { span: $span:expr, $($field_name:ident : $field_value:expr),+ }) => (
        $ctx.arena.alloc(StmtKind::$variant { span: $span, $($field_name: ast_element!($ctx, $field_value)),* })?
    );
}