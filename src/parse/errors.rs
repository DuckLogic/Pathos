use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt::{self, Display, Formatter, Debug};
use std::ops::{Range};
use std::num::{NonZeroU64};

use fixedbitset::FixedBitSet;

use crate::alloc::AllocError;
use crate::ast::Span;
use crate::lexer::StringError;

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
    /// The character boundaries of a line,
    /// implemented as a bitset (only present if the line isn't entirely ASCII)
    character_boundaries: Option<FixedBitSet>
}
impl LineCache {
    fn first_line() -> LineCache {
        LineCache {
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
    fn num_bytes(&self) -> u64 {
        self.end - self.start
    }
    /// Extend this line's length by the specified amount
    fn extend_len(&mut self, amount: u64) {
        assert!(self.end >= self.start);
        self.end = self.end.checked_add(amount).expect("u64 overflow");
    }
    fn count_chars(&self) -> usize {
        match self.character_boundaries {
            Some(ref boundaries) => boundaries.count_ones(0..self.num_bytes() as usize),
            None => self.num_bytes() as usize // ascii only
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
                assert_eq!(boundaries.len() as u64, self.num_bytes() + 1, "Unexpected boundaries length for line #{}: {:?}", line_number, self);
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
    #[inline]
    pub fn num_lines(&self) -> usize {
        self.lines.len()
    }
    pub fn reset(&mut self)  {
        self.lines.clear();
        self.lines.push(LineCache::first_line());
    }
    /// Resolve a byte-based index into a detailed location
    pub fn resolve_location(&self, byte_index: u64) -> DetailedLocation {
        let line_index = match self.lines.binary_search_by_key(&byte_index, |line| line.start) {
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
        if byte_index == self.total_bytes() {
            return DetailedLocation {
                line_number, column: line.count_chars() as u64
            };
        }
        line.resolve_index(line_number, byte_index)
    }
    pub fn resolve_span(&self, span: Span) -> DetailedSpan {
        DetailedSpan {
            start: self.resolve_location(span.start),
            end: self.resolve_location(span.end)
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
            let new_len = last_line.num_bytes() as usize;
            if let Some(ref mut boundaries) = last_line.character_boundaries {
                boundaries.grow(new_len + 1);
                boundaries.set(new_len - 1, true);
                boundaries.set(new_len, true)
            }
        }
        self.lines.push(LineCache {
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
        let old_len = last_line.num_bytes() as usize;
        last_line.extend_len(text.len() as u64);
        let new_len = last_line.num_bytes() as usize;
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
#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    InvalidToken,
    InvalidExpression,
    AllocationFailed,
    UnexpectedEof,
    UnexpectedEol,
    UnexpectedToken,
    InvalidString(StringError),
}
#[derive(Copy, Clone, Debug)]
pub enum MaybeSpan {
    Missing,
    Detailed(DetailedSpan),
    Regular(Span)
}
#[derive(Debug)]
struct ParseErrorInner {
    /// The span of the source location
    ///
    /// There are only two instances where this can be `None`:
    /// 1. Out of memory errors
    /// 2. A ParseVisitError
    span: MaybeSpan,
    expected: Option<String>,
    actual: Option<String>,
    kind: ParseErrorKind,
    backtrace: Backtrace
}

pub struct ParseError(Box<ParseErrorInner>);
impl ParseError {
    pub fn with_line_numbers(mut self, line_number_tracker: &LineTracker) -> Self {
        self.0.span = match self.0.span {
            MaybeSpan::Missing => MaybeSpan::Missing,
            MaybeSpan::Detailed(already_detailed) => MaybeSpan::Detailed(already_detailed),
            MaybeSpan::Regular(span) => MaybeSpan::Detailed(line_number_tracker.resolve_span(span))
        };
        self
    }
    /// Give additional context on the type of item that was "expected"
    #[cold]
    pub fn with_expected_msg<T: ToString>(mut self, msg: T) -> Self {
        self.0.expected = Some(msg.to_string());
        self
    }
    #[inline]
    pub fn builder(span: Span, kind: ParseErrorKind) -> ParseErrorBuilder {
        ParseErrorBuilder(ParseErrorInner {
            span: MaybeSpan::Regular(span), kind,
            expected: None, actual: None,
            backtrace: Backtrace::disabled() // NOTE: Actual capture comes later
        })
    }
    /// The span of this error, if any
    #[inline]
    pub fn span(&self) -> MaybeSpan {
        self.0.span
    }
}
impl From<AllocError> for ParseError {
    #[cold]
    #[track_caller]
    fn from(_cause: AllocError) -> Self {
        /*
         * TODO: Handle this without allocating.
         * Its bad to allocate memory for the error value
         * even though we have an 'out of memory' condition.
         * Realistically though, we'll probably only encounter
         * OOM if the counter hits the internal limit.
         */
        ParseError(Box::new(ParseErrorInner {
            span: MaybeSpan::Missing,
            expected: None,
            actual: None,
            kind: ParseErrorKind::AllocationFailed,
            backtrace: Backtrace::capture()
        }))
    }
}
impl Debug for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ParseError");
        match self.0.span {
            MaybeSpan::Detailed(span) => {
                debug.field("span", &format_args!("{}", span));
            },
            MaybeSpan::Regular(span) => {
                debug.field("span", &format_args!("{}", span));
            }
            MaybeSpan::Missing => {}
        }
        debug.field("expected", &self.0.expected)
            .field("actual", &self.0.actual)
            .field("kind", &self.0.kind)
            .field("backtrace", &format_args!("{}", self.0.backtrace))
            .finish()
    }
}
impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0.kind {
            ParseErrorKind::InvalidToken => {
                f.write_str("Invalid token")?;
            },
            ParseErrorKind::InvalidExpression => {
                f.write_str("Invalid expression")?;
            },
            ParseErrorKind::AllocationFailed => {
                f.write_str("Allocation failed")?;
            }
            ParseErrorKind::UnexpectedEof => {
                f.write_str("Unexpected EOF")?;
            },
            ParseErrorKind::UnexpectedEol => {
                f.write_str("Unexpected end of line")?;
            }
            ParseErrorKind::UnexpectedToken => {
                f.write_str("Unexpected token")?;
            }
            ParseErrorKind::InvalidString(ref cause) => {
                write!(f, "Invalid string ({})", cause)?;
            },
        }
        match (&self.0.expected, &self.0.actual) {
            (Some(ref expected), Some(ref actual)) => {
                write!(f, ": Expected {}, but got {}", expected, actual)?;
            },
            (Some(ref expected), None) => {
                write!(f, ": Expected {}", expected)?;
            },
            (None, Some(ref actual)) => {
                write!(f, ": Got {}", actual)?;
            },
            (None, None) => {}
        }
        match self.0.span {
            MaybeSpan::Regular(span) => {
                write!(f, " @ {}", span)?;
            },
            MaybeSpan::Detailed(span) => {
                write!(f, " @ {}", span)?;
            },
            MaybeSpan::Missing => {}
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.0.kind {
            ParseErrorKind::InvalidString(ref cause) => Some(cause),
            _ => None
        }
    }

    #[inline]
    fn backtrace(&self) -> Option<&Backtrace> {
        Some(&self.0.backtrace)
    }

}

pub struct ParseErrorBuilder(ParseErrorInner);

impl ParseErrorBuilder {
    #[inline]
    pub fn expected(mut self, f: impl ToString) -> Self {
        self.0.expected = Some(f.to_string());
        self
    }
    #[inline]
    pub fn actual(mut self, f: impl ToString) -> Self {
        self.0.actual = Some(f.to_string());
        self
    }
    #[inline]
    pub fn build(mut self) -> ParseError {
        ParseError(Box::new(ParseErrorInner {
            span: self.0.span,
            kind: self.0.kind,
            expected: self.0.expected.take(),
            actual: self.0.actual.take(),
            backtrace: Backtrace::capture()
        }))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn single_line() {
        // Test ASCII
        assert_eq!(LineTracker::from_text("").lines[0].num_bytes(), 0);
        let mut tracker = LineTracker::from_text("All cows eat grass");
        assert_eq!(tracker.lines[0], LineCache {
            start: 0,
            end: 18,
            character_boundaries: None
        });
        assert_eq!(tracker.resolve_location(4), DetailedLocation {
            line_number: LineNumber::ONE,
            column: 4
        });
        // Test unicode
        tracker = LineTracker::from_text("Unicode: \u{E2}ll \u{14D}ows eat gr\u{1CE}ss");
        assert_eq!(tracker.lines[0].start, 0);
        let difference_between_chars_and_bytes = Cell::new(0);
        let verify_char_position = |char_index: u64| {
            assert_eq!(
                tracker.resolve_location(char_index + difference_between_chars_and_bytes.get()),
                DetailedLocation {
                    line_number: LineNumber::ONE,
                    column: char_index
                }
            );
        };
        // 3rd char is still ASCII
        assert_eq!(difference_between_chars_and_bytes.get(), 0);
        verify_char_position(3);
        // 9th character is unicode, but it *starts* at byte-index 9 (because everything before it is ASCII)
        verify_char_position(9);
        // Character 9 is two bytes, offsetting all future characters by one
        difference_between_chars_and_bytes.update(|i| i + 1);
        verify_char_position(10);
        verify_char_position(11);
        // Character 13 is also unicode and its two bytes
        verify_char_position(13);
        difference_between_chars_and_bytes.update(|i| i + 1);
        verify_char_position(14);
        // Character 24 is also unicode and its two bytes
        verify_char_position(24);
        difference_between_chars_and_bytes.update(|i| i + 1);
        verify_char_position(25);
        verify_char_position(26);
    }
    fn verify(text: &str) {
        let tracker = LineTracker::from_text(text);
        let mut last_line = 0;
        for (line_index, line) in text.split('\n').enumerate() {
            let line_number = LineNumber::from_index(line_index);
            for (char_offset, (offset, _)) in line.char_indices().enumerate() {
                assert_eq!(tracker.resolve_location((last_line + offset) as u64), DetailedLocation {
                    column: char_offset as u64,
                    line_number
                }, "Mismatched locations for {} at {} in line #{}", char_offset, offset, line_number);
            }
            assert_eq!(tracker.resolve_location(last_line as u64 + line.len() as u64), DetailedLocation {
                line_number, column: line.chars().count() as u64
            });
            last_line += line.len() + 1;
        }
    }
    #[test]
    fn test_multiline() {
        verify("\nAll cows eat grass");
        verify("All cows\n eat grass");
        verify("All cows\n eat grass\n");
        verify("Unicode: \u{E2}ll \u{14D}\nows eat gr\u{1CE}ss");
        verify("Unicode: \u{E2}ll \u{14D}\nows eat gr\u{1CE}ss\n");
        verify("Unicode: \u{E2}ll\n \u{14D}ows eat gr\u{1CE}ss\n");
    }
}
