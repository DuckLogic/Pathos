//! Handling of identifiers

use crate::ast::{Span, Spanned};
use std::fmt::{Debug, Formatter, Display};
use std::hash::{Hash, Hasher, BuildHasher};
use std::fmt;
use std::ops::Deref;
use crate::alloc::{Allocator, AllocError};
use std::borrow::Borrow;
use hashbrown::HashMap;

/// An identifier, with a specific source location.
///
/// Contains a [Span], alongside a plain [Symbol]
#[derive(Copy, Clone)]
pub struct Ident<'a> {
    pub symbol: Symbol<'a>,
    pub span: Span,
}
impl<'a> Ident<'a> {
    /// Access the raw text of this identifier
    #[inline]
    pub fn text(&self) -> &'a str {
        self.symbol.text()
    }
}
impl<'a> Deref for Ident<'a> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.text()
    }
}
impl Eq for Ident<'_> {}
impl PartialEq for Ident<'_> {
    #[inline]
    fn eq(&self, other: &Ident) -> bool {
        self.symbol == other.symbol
    }
}
impl PartialEq<str> for Ident<'_> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.text() == other
    }
}
impl Hash for Ident<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash_code(state)
    }
}
impl<'a> Spanned for Ident<'a> {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}
impl Debug for Ident<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} @ {}", self.name, self.span)
    }
}
impl Display for Ident<'_> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}
/// A unique identifier, that doesn't correspond to a location in the source code.
///
/// In contrast to [Ident], this doesn't have a [Span].
///
/// It is basically just a (unique) pointer to a string.
///
/// These are interned separately from an [Ident]
/// and pointers to them should be unique
/// within a given compilation unit.
#[derive(Copy, Clone)]
pub struct Symbol<'a>(&'a SymbolInner<'a>);
struct SymbolInner<'a> {
    text: &'a str,
    hash: u64
}
impl<'a> Symbol<'a> {
    /// The underlying text of this symbol
    #[inline]
    pub fn text(&self) -> &'a str {
        self.0.text
    }
}
impl<'a> Borrow<str> for Symbol<'a> {
    #[inline]
    fn borrow(&self) -> &str {
        self.text()
    }
}
impl<'a> AsRef<str> for Symbol<'a> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.text()
    }
}
impl<'a> Borrow<str> for Ident<'a> {
    #[inline]
    fn borrow(&self) -> &str {
        self.text()
    }
}
impl<'a> AsRef<str> for Ident<'a> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.text()
    }
}
impl Hash for Symbol<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.hash);
    }
}
impl Debug for Symbol<'_> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}
impl Display for Symbol<'_> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}
impl<'a> PartialEq<str> for Symbol<'a> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.text() == other
    }
}
impl<'a> PartialEq<Ident<'a>> for Symbol<'a> {
    #[inline]
    fn eq(&self, other: &Ident<'a>) -> bool {
        self == other.symbol
    }
}
impl<'a> PartialEq<Symbol<'a>> for Ident<'a> {
    #[inline]
    fn eq(&self, other: &Symbol<'a>) -> bool {
        self.symbol == other
    }
}
impl<'a> Eq for Symbol<'a> {}
impl<'a> PartialEq for Symbol<'a> {
    #[inline]
    fn eq(&self, other: &Symbol<'a>) -> bool {
        let identical = std::ptr::eq(self.0, other.0);
        debug_assert_eq!(
            identical,
            self.text() == other.text(),
            "Pointer equality and value equality gave different results for {:?} and {:?}",
            self, other
        );
        identical
    }
}
/// A set of interned symbols, used to ensure that [Symbol]s are unique
pub struct SymbolTable<'a> {
    alloc: &'a Allocator,
    map: SymbolMap<'a, ()>
}
impl<'a> SymbolTable<'a> {
    #[inline]
    pub fn new(alloc: &'a Allocator) -> Self {
        SymbolTable { alloc, map: SymbolMap::new() }
    }
    /// Allocate a symbol with the specified text,
    /// reusing any existing memory if possible.
    ///
    /// Does not check for validity.
    #[inline]
    pub fn alloc(&mut self, s: &str) -> Result<Symbol<'a>, AllocError> {
        use hashbrown::hash_map::RawEntryMut;
        match self.map.entry(s) {
            RawEntryMut::Occupied(entry) => Ok(*entry.into_key()),
            RawEntryMut::Vacant(entry) => {
                let sym = self._actually_alloc()?;
                entry.insert_hashed_nocheck(sym.0.hash, sym, ());
                Ok(sym)
            }
        }
    }
    #[cold]
    #[inline(never)]
    fn _actually_alloc(&self, s: &str) -> Result<Symbol<'a>, AllocError> {
        let text = self.alloc.alloc_str(s)?;
        Ok(self.alloc.alloc(Symbol {

        }))
    }
}

/// A map of [Symbol]s to values
///
/// This takes advantage of the fact that symbols have pre-computed hashes,
/// and can use pointer-equality.
#[derive(Default, Clone, Debug)]
pub struct SymbolMap<'a, V>(::hashbrown::HashMap<Symbol<'a>, V>);
impl<'a, V> SymbolMap<'a, V> {
    #[inline]
    pub fn new() -> Self {
        SymbolMap(HashMap::new())
    }
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        SymbolMap(HashMap::with_capacity(capacity))
    }
    #[inline]
    pub fn from_raw_map(raw: ::hashbrown::HashMap<Symbol<'a>, V>) -> Self {
        SymbolMap(raw)
    }
    #[inline]
    pub fn get<K: SymbolKey<'a>>(&self, key: K) -> Option<&V> {
        self.0.get::<str>(&key.as_ref())
    }
    #[inline]
    pub fn entry<'m, K: SymbolKey<'a>>(&mut self, key: K) -> ::hashbrown::hash_map::RawEntryMut<'m, Symbol<'a>, V, ::hashbrown::hash_map::DefaultHashBuilder> {
        let hash = key.hash_code(self.0.hasher());
        self.0.raw_entry_mut()
            .from_hash(hash, |other_key| other_key.eq(&key))
    }
    #[inline]
    pub fn as_raw_map(&self) -> &'_ ::hashbrown::HashMap<&'a Symbol<'a>, V> {
        &self.0
    }
}

/// A key into a [SymbolMap] or [SymbolTable]
pub trait SymbolKey<'a>: AsRef<str> + Copy {
    fn into_symbol(self) -> Option<Symbol<'a>>;
    /// Compute the hash of the key,
    /// using the specified hasher
    ///
    /// This may return pre-computed results
    fn hash_code<H: BuildHasher>(&self, hasher: &H) -> u64;
    fn matches_symbol(&self, s: Symbol<'a>) -> bool;
}
impl<'a, 'b> SymbolKey<'a> for &'b str {
    #[inline]
    fn into_symbol(self) -> Option<Symbol<'a>> {
        None
    }
    #[inline]
    fn hash_code<H: BuildHasher>(&self, hasher: &H) -> u64 {
        let mut state = hasher.build_hasher();
        self.as_ref().hash(&mut state);
        state.finish()
    }
    #[inline]
    fn matches_symbol(&self, s: Symbol<'a>) -> bool {
        s.text() == self
    }
}
impl<'a> SymbolKey<'a> for Ident<'a> {
    #[inline]
    fn into_symbol(self) -> Option<Symbol<'a>> {
        Some(self.symbol)
    }
    #[inline]
    fn hash_code<H: BuildHasher>(&self, _hasher: &H) -> u64 {
        self.symbol.0.hash
    }
    #[inline]
    fn matches_symbol(&self, s: Symbol<'a>) -> bool {
        s == *self
    }
}
impl<'a> SymbolKey<'a> for Symbol<'a> {
    #[inline]
    fn into_symbol(self) -> Option<Symbol<'a>> {
        Some(self)
    }
    #[inline]
    fn hash_code<H: BuildHasher>(&self, _hasher: &H) -> u64 {
        self.0.hash
    }
    #[inline]
    fn matches_symbol(&self, s: Symbol<'a>) -> bool {
        s == *self
    }
}