use ariadne::{Report, ReportKind, Label};

use super::ParseError;
use crate::parse::errors::{ParseErrorKind, MaybeSpan};
use crate::lexer::{LineTracker, StringError, StringErrorKind, LexError};
use crate::ast::Span;
use std::error::Error;

#[derive(Debug, Copy, Clone)]
pub struct FancySpan {
    start_char: usize,
    end_char: usize
}
impl ariadne::Span for FancySpan {
    type SourceId = ();

    #[inline]
    fn source(&self) -> &Self::SourceId {
        &()
    }

    #[inline]
    fn start(&self) -> usize {
        self.start_char as usize
    }

    #[inline]
    fn end(&self) -> usize {
        self.end_char as usize
    }
}
impl Span {
    fn resolve_fancy(&self, tracker: &LineTracker) -> FancySpan {
        let start = tracker.char_index(tracker.resolve_position(self.start));
        let end = tracker.char_index(tracker.resolve_position(self.end));
        FancySpan { start_char: start as usize, end_char: end as usize }
    }
}
impl MaybeSpan {
    fn resolve_fancy(&self, tracker: &LineTracker) -> Option<FancySpan> {
        match *self {
            MaybeSpan::Missing => None,
            MaybeSpan::Detailed(detailed) => {
                Some(FancySpan {
                    start_char: tracker.char_index(detailed.start) as usize,
                    end_char: tracker.char_index(detailed.end) as usize
                })
            }
            MaybeSpan::Regular(regular) => {
                MaybeSpan::Detailed(tracker.resolve_span(regular)).resolve_fancy(tracker)
            }
        }
    }
}
pub struct FancyErrorContext<'a> {
    pub tracker: &'a LineTracker,
    pub current_file: &'a ariadne::Source
}
impl FancyErrorContext<'_> {
    pub fn ugly_error(&self, offset: usize, msg: impl ToString) -> Report<FancySpan> {
        Report::<FancySpan>::build(ReportKind::Error, (), offset)
            .with_message(msg)
            .finish()
    }
    pub fn report_error(&self, e: &impl FancyErrorTarget) -> Report<FancySpan> {
        e.build_fancy_report(self)
    }
}

pub trait FancyErrorTarget: Error {
    fn is_alloc_failed(&self) -> bool;
    fn build_fancy_report(&self, ctx: &FancyErrorContext) -> Report<FancySpan> {
        if self.is_alloc_failed() {
            return ctx.ugly_error(0, "Internal error: Allocation failed");
        }
        // NOTE: Implicit panic if no span
        let span = self.overall_span(ctx);
        let mut builder = Report::build(ReportKind::Error, (), span.start_char);
        builder = builder.with_message(self.build_fancy_message(ctx));
        for label in self.build_fancy_labels(ctx) {
            builder = builder.with_label(label);
        }
        builder.finish()
    }
    fn build_fancy_message(&self, ctx: &FancyErrorContext) -> String;
    fn build_fancy_labels(&self, ctx: &FancyErrorContext) -> Vec<Label<FancySpan>>;
    /// The overall span, broader than any specific label
    fn overall_span(&self, ctx: &FancyErrorContext) -> FancySpan;
}
impl FancyErrorTarget for LexError {
    fn is_alloc_failed(&self) -> bool {
        matches!(self, LexError::AllocFailed)
    }

    fn build_fancy_message(&self, ctx: &FancyErrorContext) -> String {
        match *self {
            LexError::InvalidToken { span: _ } => "Invalid token".to_string(),
            LexError::AllocFailed => unreachable!(),
            LexError::InvalidString(ref cause) => format!("String failed to parse: {}", cause.build_fancy_message(ctx))
        }
    }

    fn build_fancy_labels(&self, ctx: &FancyErrorContext) -> Vec<Label<FancySpan>> {
        match *self {
            LexError::InvalidToken { span } => {
                vec![Label::new(span.resolve_fancy(ctx.tracker))
                    .with_message("This text")]
            }
            LexError::AllocFailed => unreachable!(),
            LexError::InvalidString(ref cause) => cause.build_fancy_labels(ctx)
        }
    }

    fn overall_span(&self, ctx: &FancyErrorContext) -> FancySpan {
        match *self {
            LexError::InvalidToken { span } => span.resolve_fancy(ctx.tracker),
            LexError::AllocFailed => unreachable!(),
            LexError::InvalidString(ref cause) => cause.overall_span(ctx)
        }
    }
}
impl FancyErrorTarget for ParseError {
    fn is_alloc_failed(&self) -> bool {
        matches!(self.0.kind, ParseErrorKind::AllocationFailed)
    }

    fn build_fancy_message(&self, ctx: &FancyErrorContext) -> String {
        let mut message = match self.0.kind {
            ParseErrorKind::InvalidToken => {
                "Invalid token.".to_string()
            }
            ParseErrorKind::InvalidExpression => {
                "Invalid expression.".to_string()
            }
            ParseErrorKind::AllocationFailed => unreachable!(), // checked for above
            ParseErrorKind::UnexpectedEof => {
                "Unexpected end of file.".to_string()
            }
            ParseErrorKind::UnexpectedEol => {
                "Unexpected end of line.".to_string()
            }
            ParseErrorKind::UnexpectedToken => {
                "Unexpected token.".to_string()
            }
            ParseErrorKind::InvalidString(ref s) => s.build_fancy_message(ctx),
        };
        if !message.ends_with('.') {
            message.push('.');
        }
        if let (&Some(ref expected), &None) = (&self.0.expected, &self.0.actual) {
            message.push_str(" Expected");
            message.push_str(&**expected);
            message.push('.');
        }
        message
    }

    fn build_fancy_labels(&self, ctx: &FancyErrorContext) -> Vec<Label<FancySpan>> {
        let label_message = if let Some(ref expected) = self.0.expected {
            expected.clone()
        } else if let Some(ref actual) = self.0.actual {
            actual.clone()
        } else {
            match self.0.kind {
                ParseErrorKind::InvalidToken => "This text".to_string(),
                ParseErrorKind::InvalidExpression => "This expression".to_string(),
                ParseErrorKind::AllocationFailed => unreachable!(),
                ParseErrorKind::UnexpectedEof | ParseErrorKind::UnexpectedEol => return vec![],
                ParseErrorKind::UnexpectedToken => "This token".to_string(),
                ParseErrorKind::InvalidString(ref cause) => return cause.build_fancy_labels(ctx),
            }
        };
        vec![Label::new(self.overall_span(ctx)).with_message(label_message)]
    }

    fn overall_span(&self, ctx: &FancyErrorContext) -> FancySpan {
        match self.0.kind {
            ParseErrorKind::AllocationFailed => unreachable!(),
            ParseErrorKind::InvalidToken |
            ParseErrorKind::InvalidExpression |
            ParseErrorKind::UnexpectedEof |
            ParseErrorKind::UnexpectedEol |
            ParseErrorKind::UnexpectedToken => self.0.span.resolve_fancy(ctx.tracker).unwrap(),
            ParseErrorKind::InvalidString(ref cause) => cause.overall_span(ctx)
        }
    }
}
impl FancyErrorTarget for StringError {
    fn is_alloc_failed(&self) -> bool {
        matches!(self.kind, StringErrorKind::AllocFailed)
    }

    fn build_fancy_message(&self, _ctx: &FancyErrorContext) -> String {
       match self.kind {
            StringErrorKind::MissingEnd => {
                "Missing closing quote".to_string()
            }
            StringErrorKind::ForbiddenNewline => {
                "Unescaped newlines are forbidden".to_string()
            },
            StringErrorKind::InvalidEscape { c } => {
                format!("Invalid escape char: {:?}", c)
            }
            StringErrorKind::InvalidNamedEscape => {
                "Invalid named escape".to_string()
            }
            StringErrorKind::UnsupportedNamedEscape => {
                "Named escapes are unsupported".to_string()
            }
            StringErrorKind::AllocFailed => unreachable!()
        }
    }

    fn build_fancy_labels(&self, ctx: &FancyErrorContext) -> Vec<Label<FancySpan>> {
        let mut label = Label::new(self.span.unwrap().resolve_fancy(ctx.tracker));
        match self.kind {
            StringErrorKind::MissingEnd | StringErrorKind::ForbiddenNewline => {
                return vec![] // no labels
            }
            StringErrorKind::InvalidEscape { .. } | StringErrorKind::InvalidNamedEscape | StringErrorKind::UnsupportedNamedEscape => {
                label = label.with_message("At this escape")
            },
            StringErrorKind::AllocFailed => unreachable!()
        }
        vec![label]
    }

    fn overall_span(&self, ctx: &FancyErrorContext) -> FancySpan {
        let original_span = match self.entire_string_span {
            None => self.span.unwrap(),
            Some(val) => val
        };
        original_span.resolve_fancy(ctx.tracker)
    }
}