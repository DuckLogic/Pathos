use std::fmt::{Debug, Display};
use std::ops::Deref;

use crate::alloc::AllocError;
use crate::ast::{Span};
use crate::errors::{SpannedError,};
use std::marker::PhantomData;

/// The type identifies the type (or kind) of token
pub trait TokenKind: Copy + Debug + Display + 'static {}
/// A specific type of token given by the lexer.
pub trait Token<'a>: Copy + Display + Debug + PartialEq + 'a {
    type Kind: TokenKind;
    fn kind(&self) -> Self::Kind;
    /// Determine whether this token is a newline token.
    ///
    /// This is only true in languages that give special meaning to newlines,
    /// as opposed to treating it as whitespace.
    fn is_newline(&self) -> bool;
}
#[derive(Copy, Clone, Debug)]
pub struct SpannedToken<'a, T: Token<'a>> {
    pub span: Span,
    pub kind: T,
    pub marker: PhantomData<&'a T>
}
impl<'a, T: Token<'a>> Deref for SpannedToken<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.kind
    }
}
impl<'a, T: Token<'a>> PartialEq<T> for SpannedToken<'a, T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.kind == *other
    }
}

pub trait Lexer<'a>: Debug {
    type Error: LexerError;
    type Token: Token<'a>;
    fn original_text(&self) -> &str;
    fn current_span(&self) -> Span;
    fn try_next(&mut self) -> Result<Option<Self::Token>, Self::Error>;
}

pub trait LexerError: SpannedError + 'static {
    fn upcast(&self) -> &(dyn std::error::Error + 'static);
    /// Cast this error into an allocation failure (if any)
    fn cast_alloc_failed(&self) -> Option<&AllocError>;
}