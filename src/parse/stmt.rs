use crate::parse::PythonParser;
use crate::ast::tree::{Stmt, StmtKind, Alias, ModulePath};
use crate::lexer::Token;
use crate::ParseError;
use crate::ast::{Spanned, Span};
use crate::parse::parser::{ParseSeperated, Parser};
/*
 * Override the global `std::alloc::Vec` with our `crate::alloc::Vec`.
 * This is needed because we use a limiting, arena-allocator.
 */
use crate::alloc::Vec;
use std::hash::Hash;
use std::fmt::Debug;

impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    pub fn statement(&mut self) -> Result<Stmt<'a>, ParseError> {
        let stmt = match self.parser.peek() {
            Some(Token::Assert) => self.parse_assertion()?,
            Some(Token::Pass) => {
                let span = self.parser.expect(Token::Pass)?.span;
                self.stmt(StmtKind::Pass { span })?
            },
            Some(Token::Del) => {
                let start_span = self.parser.expect(Token::Del)?.span.start;
                let targets = self.parse_target_list(
                    &mut |parser| parser.parser.is_newline(),
                    &"newline"
                )?;
                self.stmt(StmtKind::Delete {
                    span: Span { start: start_span, end: targets.span().unwrap().start },
                    targets: targets.into_slice()?
                })?
            },
            Some(Token::Return) => {
                let start_span = self.parser.expect(Token::Return)?.span;
                let exprs  = self.parse_expression_list(
                    true, |parser: &Parser| parser.is_newline(),
                    &"newline"
                )?;
                self.stmt(StmtKind::Return {
                    span: Span { start: start_span.start, end: match exprs.span() {
                        Some(span) => span.end,
                        None => start_span.end
                    } },
                    value: exprs.with_implicit_tuple()?
                })?
            }
            Some(Token::Yield) => {
                let expr = self.parse_yield_expr(|parser: &Parser| parser.is_newline())?;
                self.stmt(StmtKind::Expr {
                    value: expr, span: expr.span()
                })?
            },
            Some(Token::Raise) => {
                let mut span = self.parser.expect(Token::Raise)?.span;
                let expr;
                let from;
                if !self.parser.is_newline() {
                    expr = Some(self.expression()?);
                    span.end = expr.as_ref().unwrap().span().end;
                    if let Some(Token::From) = self.parser.peek() {
                        from = Some(self.expression()?);
                        span.end = from.as_ref().unwrap().span().end;
                    } else {
                        from = None
                    }
                } else {
                    expr = None;
                    from = None;
                }
                self.arena.alloc(StmtKind::Raise {
                    cause: from, span, exc: expr
                })?
            }
            Some(Token::Break) => {
                let span = self.parser.expect(Token::Break)?.span;
                self.arena.alloc(StmtKind::Break { span })?
            },
            Some(Token::Continue) => {
                let span = self.parser.expect(Token::Continue)?.span;
                self.arena.alloc(StmtKind::Continue { span })?
            },
            Some(Token::Import) => {
                let span = self.parser.expect(Token::Continue)?.span;
                let mut aliases = Vec::new(self.arena);
                if self.parser.is_newline() {
                    return Err(self.parser.unexpected(&"Expected at least one import"))
                }
                while !self.parser.is_newline() {
                    aliases.push(self.parse_import_alias(|p| p.parse_module_path())?)?;
                }
                assert!(!aliases.is_empty());
                self.arena.alloc(StmtKind::Import {
                    span, names: aliases.into_slice()
                })?
            }
            _ => return Err(self.parser.unexpected(&"a statement"))
        };
        match self.parser.peek() {
            Some(Token::Newline) => {
                self.parser.skip()?;
                Ok(stmt)
            },
            None if self.parser.is_end_of_file() => {
                Ok(stmt)
            },
            None => {
                assert_eq!(self.parser.look_behind(1), Some(Token::Newline));
                unreachable!("Already consumed newline")
            }
            _ => Err(self.parser.unexpected(&"an end of line"))
        }
    }
    fn parse_import_alias<N, F>(&mut self, mut inner_parser: F) -> Result<Alias<'a, N>, ParseError>
        where F: FnMut(&mut Self) -> Result<N, ParseError>, N: Spanned + Eq + Hash + Clone + Debug {
        let start = self.parser.current_span().end;
        let name = inner_parser(&mut *self)?;
        let renamed = if let Some(Token::As) = self.parser.peek() {
            self.parser.skip()?;
            Some(self.parse_ident()?)
        } else {
            None
        };
        let end = match renamed {
            Some(ref name) => name.span.end,
            None => name.span().end
        };
        Ok(Alias {
            span: Span { start, end }, name, alias: renamed
        })
    }
    fn parse_module_path(&mut self) -> Result<ModulePath<'a>, ParseError> {
        const EXPECTED_MSG: &str = "a module path";
        let mut items = Vec::new(self.arena);
        if !matches!(self.parser.peek(), Some(Token::Ident(_))) {
            return Err(self.parser.unexpected(&EXPECTED_MSG))
        }
        let mut iter = ParseSeperated::new(
            self, |parser| parser.parse_ident(),
            Token::Period, |parser: &Parser| !matches!(parser.peek(), Some(Token::Ident(_))),
            false
        );
        while let Some(item) = iter.next().transpose()? {
            items.push(item)?;
        }
        Ok(ModulePath {
            parts: &*items.into_slice()
        })
    }
    #[inline]
    fn stmt(&mut self, stmt: StmtKind<'a>) -> Result<Stmt<'a>, ParseError> {
        Ok(&*self.arena.alloc(stmt)?)
    }
    fn parse_assertion(&mut self) -> Result<Stmt<'a>, ParseError> {
        let start = self.parser.expect(Token::Assert)?.span.start;
        let condition = self.expression()?;
        let message = if let Some(Token::Comma) = self.parser.peek() {
            self.parser.skip()?;
            Some(self.expression()?)
        } else { None };
        let end = match message {
            Some(expr) => expr.span().end,
            None => condition.span().end
        };
        Ok(&*self.arena.alloc(StmtKind::Assert {
            span: Span { start, end }, msg: message, test: condition
        })?)
    }

}