use std::num::NonZeroU64;
use std::fmt::{self, Formatter, Display, Debug};

use fixedbitset::FixedBitSet;
use std::ops::Range;
use crate::ast::{Position, Span};

#[derive(Copy, Clone, Eq, PartialOrd, PartialEq, Ord)]
pub struct LineNumber(NonZeroU64);
impl LineNumber {
    /// The first line
    pub const ONE: LineNumber = LineNumber(NonZeroU64::new(1).unwrap());

    #[inline]
    #[track_caller]
    pub fn from_index(v: usize) -> LineNumber {
        LineNumber(NonZeroU64::new((v as u64).checked_add(1).expect("Overflow line number: u64::MAX + 1")).unwrap())
    }
    #[inline]
    #[track_caller]
    pub fn new(v: u64) -> LineNumber {
        LineNumber(NonZeroU64::new(v).expect("Invalid line number: 0"))
    }
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.get()
    }
}
impl Display for LineNumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}
impl Debug for LineNumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}
/// A detailed location in the source
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DetailedLocation {
    /// The one-based line number
    pub line_number: LineNumber,
    /// The zero-based offset of the character within the line
    pub column: u64,
}
impl Display for DetailedLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line_number, self.column)
    }
}
/// A more detailed version of [Span]
///
/// Contains both line number and column information
#[derive(Copy, Clone)]
pub struct DetailedSpan {
    pub start: DetailedLocation,
    pub end: DetailedLocation
}
#[derive(Debug, PartialEq)]
struct LineCache {
    /// The start index of the line
    start: u64,
    /// The end index of the line (exclusive)
    ///
    /// This should include the `\n` at the end of the line,
    /// unless this is the end of the file (then the whole thing should be empty)
    end: u64,
    /// The number of chars before the start of this line
    char_offset: u64,
    /// The character boundaries of a line,
    /// implemented as a bitset (only present if the line isn't entirely ASCII)
    character_boundaries: Option<FixedBitSet>
}
impl LineCache {
    fn first_line() -> LineCache {
        LineCache {
            char_offset: 0,
            start: 0,
            end: 0,
            character_boundaries: None
        }
    }
    #[inline]
    fn range(&self) -> Range<u64> {
        self.start..self.end
    }
    /// The length of the line
    #[inline]
    fn byte_len(&self) -> usize {
        (self.end - self.start) as usize
    }
    #[inline]
    fn char_len(&self) -> u64 {
        self.count_chars(self.byte_len()) as u64
    }
    /// Extend this line's length by the specified amount
    fn extend_len(&mut self, amount: u64) {
        assert!(self.end >= self.start);
        self.end = self.end.checked_add(amount).expect("u64 overflow");
    }
    fn count_chars(&self, byte_offset: usize) -> usize {
        assert!(byte_offset <= self.byte_len());
        match self.character_boundaries {
            Some(ref boundaries) => boundaries.count_ones(0..byte_offset),
            None => self.byte_len() as usize // ascii only
        }
    }
    /// Resolve a byte offset, assuming it is present in this line
    fn resolve_index(&self, line_number: LineNumber, byte_index: u64) -> DetailedLocation {
        assert!(
            self.range().contains(&byte_index),
            "Invalid byte index {} for line #{} (only valid for {}..{})",
            byte_index, line_number, self.start, self.end
        );
        let byte_offset = byte_index - self.start;
        let char_offset = match self.character_boundaries {
            Some(ref boundaries) => {
                assert_eq!(boundaries.len(), self.byte_len() + 1, "Unexpected boundaries length for line #{}: {:?}", line_number, self);
                assert!(boundaries[0], "Start of line should always be a character boundary");
                assert!(
                    boundaries[byte_offset as usize],
                    "Not on a byte offset: {}, for line #{}, starting at {} (nearby words: {:b})",
                    byte_index, line_number, self.start, boundaries.as_slice()[byte_offset as usize / 32]
                );
                boundaries.count_ones(0..byte_offset as usize) as u64
            },
            None => byte_offset // We're entirely ASCII
        };
        DetailedLocation { column: char_offset, line_number }
    }
}
#[derive(Debug)]
pub struct LineTracker {
    lines: Vec<LineCache>,
}
impl Default for LineTracker {
    fn default() -> Self {
        Self::new()
    }
}
impl LineTracker {
    /// Create a line tracker from the specified text
    pub fn from_text(text: &str) -> LineTracker {
        let mut res = LineTracker::new();
        let mut last_line_start = 0;
        for line_start in ::memchr::memchr_iter(b'\n', text.as_bytes()) {
            // Mark the line before
            res.extend_character_boundaries(&text[last_line_start..line_start]);
            res.mark_line_start(line_start as u64 + 1);
            last_line_start = line_start + 1;
        }
        res.extend_character_boundaries(&text[last_line_start..]);
        res
    }
    pub fn new() -> LineTracker {
        LineTracker { lines: vec![LineCache::first_line()] }
    }
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        self.lines.last().unwrap().end
    }
    pub fn total_chars(&self) -> u64 {
        let last = self.lines.last().unwrap();
        last.char_offset + last.count_chars(last.byte_len() as usize) as u64
    }
    #[inline]
    pub fn num_lines(&self) -> usize {
        self.lines.len()
    }
    pub fn reset(&mut self)  {
        self.lines.clear();
        self.lines.push(LineCache::first_line());
    }
    pub fn char_index(&self, location: DetailedLocation) -> u64 {
        let line = &self.lines[location.line_number.get() as usize - 1];
        line.char_offset + line.count_chars(location.column as usize) as u64 - 1
    }
    /// Resolve a byte-based index into a detailed location
    pub fn resolve_position(&self, pos: Position) -> DetailedLocation {
        let line_index = match self.lines.binary_search_by_key(&pos.0, |line| line.start) {
            Ok(index) => {
                // The index is *exactly* the start of a line
                index
            },
            Err(insertion_index) => {
                /*
                 * The index comes after the start of a line.
                 * The 'insertion_index' is where the index could be inserted to maintain sorted order.
                 * For example, with line starts [0, 2, 7, 14, 18]
                 * The value '8' would give the index '3'.
                 * Subtract one from the insertion_index to get the value before (the line's start location)
                 * NOTE: We know the subtraction cannot overflow,
                 * because index > 0 (otherwise we would've been Ok)
                 */
                insertion_index.checked_sub(1).unwrap()
            }
        };
        let line = &self.lines[line_index];
        let line_number = LineNumber::from_index(line_index);
        // Need to handle last byte specially (because line.end is exclusive)
        if pos.0 == self.total_bytes() {
            return DetailedLocation {
                line_number, column: line.count_chars(line.byte_len() as usize) as u64
            };
        }
        line.resolve_index(line_number, pos.0)
    }
    pub fn resolve_span(&self, span: Span) -> DetailedSpan {
        DetailedSpan {
            start: self.resolve_position(span.start),
            end: self.resolve_position(span.end)
        }
    }
    /// Mark an index as the start of a newline
    ///
    /// This must always be done in increasing order.
    ///
    /// This marks the index
    ///
    /// NOTE: `0` is always (implicitly)
    /// the start of a newline and does not need to be marked explicitly.
    ///
    /// Implicitly marks the end of the last line.
    pub fn mark_line_start(&mut self, line_start: u64) {
        let last_line = self.lines.last_mut().unwrap();
        assert!(
            line_start == last_line.end || line_start == last_line.end + 1,
            "Start of next line {} must == end of last line {} (or 1 past it)",
            line_start, last_line.end
        );
        if line_start == last_line.end + 1 {
            last_line.end = line_start;
            let new_len = last_line.byte_len() as usize;
            if let Some(ref mut boundaries) = last_line.character_boundaries {
                boundaries.grow(new_len + 1);
                boundaries.set(new_len - 1, true);
                boundaries.set(new_len, true)
            }
        }
        let char_offset = last_line.char_offset + last_line.char_len();
        self.lines.push(LineCache {
            char_offset,
            start: line_start, end: line_start,
            character_boundaries: None
        });
    }
    /// Mark the end of the file
    ///
    /// This sets the length of the last line
    #[inline]
    pub fn mark_eof(&mut self, file_end: u64) {
        let last_line = self.lines.last_mut().unwrap();
        assert!(file_end >= last_line.start, "EOF {} must come after last line start {}", file_end, last_line.start);
        last_line.end = file_end;
    }
    /// Mark the line's character boundaries
    ///
    /// This does nothing if the line only contains ASCII characters.
    /// In other words, it implicitly checks for ASCII-only text and ignores it.
    ///
    /// This implicitly extends the end of the line.
    pub fn extend_character_boundaries(&mut self, text: &str) {
        let last_line = self.lines.last_mut().unwrap();
        let old_len = last_line.byte_len() as usize;
        last_line.extend_len(text.len() as u64);
        let new_len = last_line.byte_len() as usize;
        if let Some(ref mut set) = last_line.character_boundaries {
            set.grow(new_len + 1);
        }
        let mut last_char_offset = 0;
        for offset in text.char_indices().skip(1)
            .map(|(offset, _)| offset)
            .chain(std::iter::once(text.len())) {
            if offset != last_char_offset + 1 && last_line.character_boundaries.is_none() {
                let mut set = FixedBitSet::with_capacity(text.len() + 1);
                set.set_range(old_len..old_len + last_char_offset + 1, true); // these are all ASCII chars
                last_line.character_boundaries = Some(set);
            }
            if let Some(ref mut set) = last_line.character_boundaries {
                set.set(old_len + offset, true);
            }
            last_char_offset = offset;
        }
    }
}
impl Debug for DetailedSpan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
impl Display for DetailedSpan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.start.line_number == self.end.line_number {
            write!(f, "{}:{}..{}", self.start.line_number, self.start.column, self.end.column)
        } else {
            write!(f, "{}..{}", self.start, self.end)
        }
    }
}
