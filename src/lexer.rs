use std::fmt::{Debug, Display, Formatter};
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
impl<'a, T: Token<'a>> Display for SpannedToken<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.kind, f)
    }
}
impl<'a, T: Token<'a>> Deref for SpannedToken<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.kind
    }
}
impl<'a, T: Token<'a>> PartialEq for SpannedToken<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
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

#[cfg(feature = "fancy-errors")]
pub trait FancyErrors: crate::errors::fancy::FancyErrorTarget {}
#[cfg(feature = "fancy-errors")]
impl<T: crate::errors::fancy::FancyErrorTarget> FancyErrors for T {}

#[cfg(not(feature = "fancy-errors"))]
pub trait FancyErrors: std::error::Error {}
#[cfg(not(feature = "fancy-errors"))]
impl<T: std::error::Error> FancyErrors for T {}

pub trait LexerError: SpannedError + FancyErrors + 'static {
    fn upcast(&self) -> &(dyn std::error::Error + 'static);
    /// Cast this error into an allocation failure (if any)
    fn cast_alloc_failed(&self) -> Option<&AllocError>;
}