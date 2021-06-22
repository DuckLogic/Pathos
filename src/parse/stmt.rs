use crate::parse::PythonParser;
use crate::ast::tree::{Stmt, StmtKind, Alias, ModulePath, RelativeModule};
use crate::lexer::Token;
use crate::ParseError;
use crate::ast::{Spanned, Span, AstNode};
use crate::parse::parser::{ParseSeperated, Parser, IParser, EndFunc, ParseSeperatedConfig};
/*
 * Override the global `std::alloc::Vec` with our `crate::alloc::Vec`.
 * This is needed because we use a limiting, arena-allocator.
 */
use crate::alloc::Vec;
use std::num::NonZeroU32;

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
                    true, EndFunc::new("newline", |parser: &Parser| parser.is_newline()),
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
                let expr = self.parse_yield_expr(EndFunc::new(
                    "newline",
                    |parser| parser.is_newline()
                ))?;
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
                let span = self.parser.expect(Token::Import)?.span;
                let mut aliases = Vec::new(self.arena);
                if self.parser.is_newline() {
                    return Err(self.parser.unexpected(&"Expected at least one import"))
                }
                loop {
                    aliases.push(self.parse_import_alias(|p| p.parse_module_path())?)?;
                    if let Some(Token::Comma) = self.parser.peek() {
                        self.parser.skip()?;
                        continue
                    } else {
                        break
                    }
                }
                assert!(!aliases.is_empty());
                self.arena.alloc(StmtKind::Import {
                    span, names: aliases.into_slice()
                })?
            },
            Some(Token::From) => {
                let start = self.parser.expect(Token::From)?.span.start;
                let relative_module = self.parse_relative_import_module()?;
                self.parser.expect(Token::Import)?;
                let (inside_parens, closing_token) = if matches!(self.parser.peek(), Some(Token::OpenParen)) {
                    self.parser.skip()?;
                    (true, Token::CloseParen)
                } else {
                    (false, Token::Newline)
                };
                let mut end = relative_module.span().end;
                let mut aliases = Vec::new(self.arena);
                let mut iter = self.parse_seperated(
                    Token::Comma, closing_token, |parser| {
                        parser.parse_import_alias(|parser| parser.parse_ident())
                    },
                    ParseSeperatedConfig {
                        allow_multi_line: inside_parens,
                        allow_terminator: inside_parens
                    }
                );
                while let Some(item) = iter.next().transpose()? {
                    end = item.span.end;
                    aliases.push(item)?;
                }
                self.stmt(StmtKind::ImportFrom {
                    module: relative_module, names: aliases.into_slice(),
                    span: Span { start, end }
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
    fn parse_relative_import_module(&mut self) -> Result<RelativeModule<'a>, ParseError> {
        let mut level = 0u32;
        let Span { start, mut end } = self.parser.current_span();
        while matches!(self.parser.peek(), Some(Token::Period)) {
            end = self.parser.current_span().end;
            self.parser.skip()?;
            level += 1;
        }
        if level == 0 {
            Ok(RelativeModule::Absolute {
                path: self.parse_module_path()?
            })
        } else {
            let relative_path = if matches!(self.parser.peek(), Some(Token::Ident(_))) {
                let p = self.parse_module_path()?;
                end = p.span().end;
                Some(p)
            } else {
                None
            };
            Ok(RelativeModule::Relative {
                relative_path, level: NonZeroU32::new(level).unwrap(),
                span: Span { start, end }
            })
        }
    }
    fn parse_import_alias<N, F>(&mut self, mut inner_parser: F) -> Result<Alias<N>, ParseError>
        where F: FnMut(&mut Self) -> Result<N, ParseError>, N: AstNode {
        let start = self.parser.current_span().end;
        let name = inner_parser(&mut *self)?;
        let renamed = if let Some(Token::As) = self.parser.peek() {
            self.parser.skip()?;
            Some(inner_parser(&mut *self)?)
        } else {
            None
        };
        let end = match renamed {
            Some(ref name) => name.span().end,
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
            Token::Period, EndFunc::new("", |parser: &Parser| !matches!(parser.peek(), Some(Token::Ident(_)))),
            ParseSeperatedConfig {
                allow_terminator: false,
                allow_multi_line: false
            }
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