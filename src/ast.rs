pub struct Span {
    start: usize,
    end: usize
}
pub struct Ident(String);
pub enum Constant {
    Tuple
}

pub trait Spanned {
    fn span(&self) -> Span;
}

include!("ast_gen.rs");