use std::error::Error;

use ariadne::{Report, ReportKind, Label};

use super::ParseError;
use crate::errors::{ParseErrorKind, InternalSpan};
use crate::errors::tracker::LineTracker;
use crate::ast::Span;

#[derive(Debug, Copy, Clone)]
pub struct FancySpan {
    pub start_char: usize,
    pub end_char: usize
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
    pub fn resolve_fancy(&self, tracker: &LineTracker) -> FancySpan {
        let start = tracker.char_index(tracker.resolve_position(self.start));
        let end = tracker.char_index(tracker.resolve_position(self.end));
        FancySpan { start_char: start as usize, end_char: end as usize }
    }
}
impl InternalSpan {
    fn resolve_fancy(&self, tracker: &LineTracker) -> Option<FancySpan> {
        match *self {
            InternalSpan::AllocationFailed => None,
            InternalSpan::Detailed { detailed, .. } => {
                Some(FancySpan {
                    start_char: tracker.char_index(detailed.start) as usize,
                    end_char: tracker.char_index(detailed.end) as usize
                })
            }
            InternalSpan::Regular(regular) => {
                InternalSpan::Detailed {
                    detailed: tracker.resolve_span(regular),
                    original: regular
                }.resolve_fancy(tracker)
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
impl FancyErrorTarget for ParseError {
    fn is_alloc_failed(&self) -> bool {
        matches!(self.0.kind, ParseErrorKind::AllocationFailed)
    }

    fn build_fancy_message(&self, ctx: &FancyErrorContext) -> String {
        let mut message = match self.0.kind {
            ParseErrorKind::UnexpectedToken => {
                "Unexpected token.".to_string()
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
            ParseErrorKind::Lexer(ref l) => {
                l.build_fancy_message(ctx)
            }
        };
        if !message.ends_with('.') {
            message.push('.');
        }
        if let (&Some(ref expected), Some(_)) = (&self.0.expected, &self.0.actual) {
            message.push_str(" Expected ");
            message.push_str(&**expected);
            message.push('.');
        }
        message
    }

    fn build_fancy_labels(&self, ctx: &FancyErrorContext) -> Vec<Label<FancySpan>> {
        let label_message = if let Some(ref actual) = self.0.actual {
            format!("Actually got {}", actual.clone())
        } else if let Some(ref expected) = self.0.expected {
            format!("Expected {}", expected.clone())
        } else {
            match self.0.kind {
                ParseErrorKind::InvalidExpression => "Actually got this expression".to_string(),
                ParseErrorKind::AllocationFailed => unreachable!(),
                ParseErrorKind::UnexpectedEof | ParseErrorKind::UnexpectedEol => return vec![],
                ParseErrorKind::UnexpectedToken => "This token".to_string(),
                ParseErrorKind::Lexer(ref cause) => return cause.build_fancy_labels(ctx),
            }
        };
        vec![Label::new(self.overall_span(ctx)).with_message(label_message)]
    }

    fn overall_span(&self, ctx: &FancyErrorContext) -> FancySpan {
        match self.0.kind {
            ParseErrorKind::AllocationFailed => unreachable!(),
            ParseErrorKind::InvalidExpression |
            ParseErrorKind::UnexpectedEof |
            ParseErrorKind::UnexpectedEol |
            ParseErrorKind::UnexpectedToken => self.0.span.resolve_fancy(ctx.tracker).unwrap(),
            ParseErrorKind::Lexer(ref cause) => cause.overall_span(ctx)
        }
    }
}