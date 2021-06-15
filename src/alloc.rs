use std::{mem, slice, cmp, str};
use std::ops::{Deref, DerefMut};
use std::alloc::{Layout, LayoutError};
use std::fmt::{self, Display, Formatter};
use std::error::Error;
use std::ptr::{NonNull};

use bumpalo::Bump;

#[macro_export(local_inner_macros)]
macro_rules! count_exprs {
    () => (0);
    ($single:expr) => (1);
    ($first:expr, $($rem:expr),*) => {
        1 + count_exprs!($($rem),*)
    };
}
#[macro_export]
macro_rules! vec {
    (in $arena:expr) => ($crate::alloc::Vec::new());
    (in $arena:expr; $($element:expr),*) => ({
        match $crate::alloc::Vec::with_capacity(
            $arena,
            $crate::count_exprs!($($element),*)
        ) {
            Ok(mut res) => {
                $(res.push($element).unwrap();)*
                Ok(res)
            },
            Err(e) => Err(e)
        }
    });
    (in $arena:expr; $($element:expr),+ ,) => ({
        vec![in $arena; $($element),*]
    })
}

/// A modified version of [std::vec::Vec]
/// for use with [Allocator]
pub struct Vec<'arena, T> {
    arena: &'arena Allocator,
    capacity: usize,
    len: usize,
    ptr: NonNull<T>,
}
/*
 * NOTE: Significant parts of this vector implementation
 * have been copied from the standard library.
 * I would suspect that this specific portion of the
 * code likely falls under their license.
 * However, this should be fine because the upstream
 * license is very liberal (MIT/Apache 2)
 */
impl<'arena, T> Vec<'arena, T> {
    #[inline]
    pub fn new(arena: &'arena Allocator) -> Self {
        Vec {
            arena, capacity: 0,
            len: 0, ptr: NonNull::dangling()
        }
    }
    #[inline]
    pub fn with_capacity(arena: &'arena Allocator, capacity: usize) -> Result<Self, AllocError> {
        let mut res = Vec::new(arena);
        res.reserve(capacity)?;
        debug_assert!(res.capacity >= capacity);
        Ok(res)
    }
    #[inline]
    pub fn push(&mut self, val: T) -> Result<(), AllocError> {
        self.reserve(1)?;
        unsafe {
            self.ptr.as_ptr().add(self.len).write(val);
            self.len += 1;
            Ok(())
        }
    }
    #[inline]
    pub fn extend_from_slice(&mut self, src: &[T]) -> Result<(), AllocError> where T: Copy {
        self.reserve(src.len())?;
        unsafe {
            self.ptr.as_ptr().add(self.len)
                .copy_from_nonoverlapping(src.as_ptr(), src.len());
        }
        self.len += src.len();
        Ok(())
    }
    #[inline]
    pub fn reserve(&mut self, amount: usize) -> Result<(), AllocError> {
        if amount > self.capacity.wrapping_sub(self.len) {
            self.grow(amount)
        } else {
            Ok(())
        }
    }
    #[cold]
    fn grow(&mut self, amount: usize) -> Result<(), AllocError> {
        debug_assert!(amount > 0);
        let required_cap = self.len.checked_add(amount).ok_or(AllocError)?;
        /*
         * This guarantees exponential growth.
         * The doubling cannot overflow
         * because `cap <= isize::MAX` and the type
         * of `cap` is `usize`.
         */
        let cap = cmp::max(self.capacity * 2, required_cap);
        let new_layout = Layout::array::<T>(cap)?;
        let new_memory = self.arena.alloc_layout(new_layout)?;
        unsafe {
            self.ptr.as_ptr().copy_to_nonoverlapping(
                new_memory.as_ptr() as *mut T,
                self.len
            );
            self.capacity = cap;
            self.ptr = new_memory.cast::<T>();
            Ok(())
        }
    }
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(
            self.ptr.as_ptr(),
            self.len
        ) }
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(
            self.ptr.as_ptr(),
            self.len
        ) }
    }
    #[inline]
    pub fn into_slice(self) -> &'arena mut [T] {
        unsafe {
            slice::from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len
            )
        }
    }
}
impl<'arena, T> Deref for Vec<'arena, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'arena, T> DerefMut for Vec<'arena, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
impl<'arena, T> Drop for Vec<'arena, T> {
    #[inline]
    fn drop(&mut self) {
        if mem::needs_drop::<T>() {
            unsafe {
                std::ptr::drop_in_place(self.as_mut_slice())
            }
        }
    }
}
/// A modified version of [std::string::String] for use
/// with this [Allocator].
///
/// This is the complement to our [Vec].
pub struct String<'arena> {
    bytes: Vec<'arena, u8>
}
impl<'arena> String<'arena> {
    #[inline]
    pub fn new(arena: &'arena Allocator) -> Self {
        String { bytes: Vec::new(arena) }
    }
    #[inline]
    pub fn with_capacity(arena: &'arena Allocator, capacity: usize) -> Result<Self, AllocError> {
        Ok(String { bytes: Vec::with_capacity(arena, capacity)? })
    }

    #[inline]
    pub fn push(&mut self, c: char) -> Result<(), AllocError> {
        let mut buf = [0u8; 4];
        self.push_str(&*c.encode_utf8(&mut buf))
    }
    #[inline]
    pub fn push_str(&mut self, src: &str) -> Result<(), AllocError> {
        self.bytes.extend_from_slice(src.as_bytes())
    }
    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe {
            std::str::from_utf8_unchecked(self.bytes.as_slice())
        }
    }
    #[inline]
    pub fn into_str(self) -> &'arena str {
        unsafe {
            std::str::from_utf8_unchecked_mut(
                self.bytes.into_slice()
            )
        }
    }
}

/// Indicating the allocator is out of memory
///
/// This can be either because the backing allocator (malloc)
/// returned an error, or the total number of allocated bytes
/// exceeded the internal limits.
///
/// It can also occur if an arithmetic overflow occurs
/// computing the size of an allocation.
///
/// This is a marker error,
/// which caries no data.
#[derive(Copy, Clone, Debug)]
pub struct AllocError;
impl Display for AllocError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("Allocation failed")
    }
}
impl Error for AllocError {}
impl From<LayoutError> for AllocError {
 #[inline]
    fn from(_cause: LayoutError) -> Self {
        AllocError
    }   
}

/// An allocator which carefully limits memory usage.
pub struct Allocator {
    limit: usize,
    arena: Bump,
}
impl Allocator {
    #[inline]
    pub fn new(arena: Bump) -> Self {
        Allocator { arena, limit: usize::MAX }
    }
    #[inline]
    pub fn set_limit(&mut self, limit: usize) -> &mut Self {
        assert!(
            limit >= self.arena.allocated_bytes(),
            "Limit {} exceeds already allocated bytes",
            limit
        );
        self.limit = limit;
        self
    }
    #[inline]
    pub fn limit(&self) -> usize {
        self.limit
    }
    #[inline]
    pub fn into_inner(self) -> Bump {
        self.arena
    }
    #[inline]
    pub fn alloc<'a, T>(&'a self, val: T) -> Result<&'a mut T, AllocError> {
        let ptr = self.alloc_layout(Layout::new::<T>())?.as_ptr() as *mut T;
        unsafe {
            ptr.write(val);
            Ok(&mut *ptr)
        }
    }
    #[inline]
    pub fn alloc_with<'a, F, T>(&'a self, func: F) -> Result<&'a mut T, AllocError>
        where F: FnOnce() -> T {
        let ptr = self.alloc_layout(Layout::new::<T>())?.as_ptr() as *mut T;
        unsafe {
            ptr.write(func());
            Ok(&mut *ptr)
        }
    }
    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        if layout.size() <= self.remaining_bytes() {
            match self.arena.try_alloc_layout(layout) {
                Ok(val) => Ok(val),
                Err(_) => Err(AllocError)
            }
        } else {
            Err(AllocError)
        }
    }
    #[inline]
    pub fn alloc_slice_copy<'a, T: Copy>(&'a self, src: &[T]) -> Result<&'a mut [T], AllocError> {
        unsafe {
            let layout = Layout::for_value(src);
            let mem = match self.arena.try_alloc_layout(layout) {
                Ok(mem) => mem.as_ptr() as *mut T,
                Err(_) => return Err(AllocError)
            };
            let len = src.len();
            mem.copy_from_nonoverlapping(src.as_ptr(), len);
            Ok(slice::from_raw_parts_mut(mem, len))
        }
    }
    #[inline]
    pub fn alloc_str<'a>(&'a self, src: &str) -> Result<&'a str, AllocError> {
        unsafe {
            let bytes = self.alloc_slice_copy(src.as_bytes())?;
            Ok(str::from_utf8_unchecked_mut(bytes))
        }
    }
    /// The remaining number of bytes before the 
    /// internal limit is reached
    ///
    /// NOTE: The underlying limit may be [usize::MAX],
    /// in which case this will return a very large number
    #[inline]
    pub fn remaining_bytes(&self) -> usize {
        self.limit - self.arena.allocated_bytes()
    }
}