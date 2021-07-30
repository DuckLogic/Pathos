use ariadne::Label;

use pathos::errors::fancy::{FancyErrorTarget, FancyErrorContext, FancySpan};

use crate::lexer::{LexError, StringError, StringErrorKind};

impl FancyErrorTarget for LexError {
    fn is_alloc_failed(&self) -> bool {
        matches!(*self, LexError::AllocFailed)
    }

    fn build_fancy_message(&self, ctx: &FancyErrorContext) -> String {
        match *self {
            LexError::InvalidToken { span: _ } => "Invalid token".to_string(),
            LexError::AllocFailed => unreachable!(),
            LexError::InvalidString(ref cause) => format!("String failed to parse: {}", cause.build_fancy_message(ctx)),
            LexError::UnclosedNesting { span: _, opening_token, expected_closing: _ } => {
                format!("Unclosed token {}", opening_token.kind)
            },
            LexError::MismatchedClosing { span: _, opening_token, expected_closing, actual_closing } => {
                format!("Closing token {} doesn't correspond to opening token {} (expected a closing {})", actual_closing.kind, opening_token.kind, expected_closing)
            }
            LexError::ClosingWithoutOpening { span: _, actual_closing } => {
                format!("Closing token {} doesn't correspond to any opening token", actual_closing)
            }
        }
    }

    fn build_fancy_labels(&self, ctx: &FancyErrorContext) -> Vec<Label<FancySpan>> {
        let tracker = ctx.tracker;
        match *self {
            LexError::InvalidToken { span } => {
                vec![Label::new(span.resolve_fancy(ctx.tracker))
                    .with_message("This text")]
            }
            LexError::AllocFailed => unreachable!(),
            LexError::InvalidString(ref cause) => cause.build_fancy_labels(ctx),
            LexError::UnclosedNesting { span: _, opening_token, expected_closing: _ } => {
                vec![Label::new(opening_token.span.resolve_fancy(tracker))
                    .with_message("This opening token")]
            }
            LexError::MismatchedClosing { span: _, opening_token, expected_closing: _, actual_closing } => {
                vec![
                    Label::new(actual_closing.span.resolve_fancy(tracker)).with_message("This closing token"),
                    Label::new(opening_token.span.resolve_fancy(tracker)).with_priority(-2).with_message("Detected opening")
                ]
            }
            LexError::ClosingWithoutOpening { span: _, actual_closing } => {
                vec![
                    Label::new(actual_closing.span.resolve_fancy(tracker)).with_message("This closing token")
                ]
            }
        }
    }

    fn overall_span(&self, ctx: &FancyErrorContext) -> FancySpan {
        match *self {
            LexError::UnclosedNesting { span, .. } |
            LexError::MismatchedClosing { span, .. } |
            LexError::ClosingWithoutOpening { span, .. } |
            LexError::InvalidToken { span } => span.resolve_fancy(ctx.tracker),
            LexError::AllocFailed => unreachable!(),
            LexError::InvalidString(ref cause) => cause.overall_span(ctx),
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
