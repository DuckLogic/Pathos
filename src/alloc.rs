use std::{slice, str};
use std::alloc::{Layout, LayoutError};
use std::fmt::{self, Display, Formatter};
use std::error::Error;
use std::ptr::NonNull;

use bumpalo::Bump;

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
impl From<std::alloc::AllocError> for AllocError {
    #[inline]
    fn from(cause: std::alloc::AllocError) -> Self {
        AllocError
    }
}
impl From<LayoutError> for AllocError {
 #[inline]
    fn from(cause: LayoutError) -> Self {
        AllocError
    }   
}

/// An allocator which carefully limits memory usage.
pub struct Allocator {
    limit: usize,
    arena: Bump
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
            Ok(self.arena.try_alloc_layout(layout)?)
        } else {
            Err(AllocError)
        }
    }
    #[inline]
    pub fn alloc_slice_copy<'a, T: Copy>(&'a self, src: &[T]) -> Result<&'a mut [T], AllocError> {
        unsafe {
            let layout = Layout::for_value(src);
            let mem = self.arena.try_alloc_layout(layout)?.as_ptr() as *mut T;
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
    pub fn remaining_bytes(&self) -> usize {
        self.limit - self.arena.allocated_bytes()
    }
}