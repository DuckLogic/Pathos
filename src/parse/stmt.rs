use crate::parse::{PythonParser, ArgumentParseOptions, SpannedToken};
use crate::ast::tree::{Stmt, StmtKind, Alias, ModulePath, RelativeModule, Expr, ExprKind};
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
            Some(Token::At) => {
                self.parse_decorated()?
            }
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
    fn parse_decorated(&mut self) -> Result<Stmt<'a>, ParseError> {
        let mut decorators = Vec::new(self.arena);
        while let Some(Token::At) = self.parser.peek() {
            self.parser.skip()?;
            decorators.push(self.expression()?)?;
            self.parser.expect(Token::Newline)?;
            self.parser.next_line()?;
        }
        let decorators = &*decorators.into_slice();
        match self.parser.peek() {
            Some(Token::Def) => {
                self.parse_function_def(decorators)
            },
            Some(Token::Async) if matches!(self.parser.look_ahead(1)?, Some(SpannedToken { kind: Token::Def, .. })) => {
                self.parse_function_def(decorators)
            },
            Some(Token::Class) => {
                self.parse_class_def(decorators)
            },
            _ => return Err(self.parser.unexpected(&"either a function or class definition"))
        }
    }
    fn parse_function_def(&mut self, decorators: &'a [Expr<'a>]) -> Result<Stmt<'a>, ParseError> {
        let start = self.parser.current_span().start;
        let is_async = match self.parser.peek() {
            Some(Token::Def) => {
                self.parser.skip()?;
                false
            },
            Some(Token::Async) => {
                self.parser.skip()?;
                self.parser.expect(Token::Def)?;
                true
            },
            _ => return Err(self.parser.unexpected(&"'def' or 'async def'"))
        };
        let name = self.parse_ident()?;
        self.parser.expect(Token::OpenParen)?;
        let arg_declarations = self.parse_argument_declarations(Token::CloseParen, ArgumentParseOptions {
            allow_type_annotations: true
        })?;
        self.parser.expect(Token::CloseParen)?;
        let return_type = if let Some(Token::CloseParen) = self.parser.peek() {
            self.parser.skip()?;
            Some(self.expression()?)
        } else { None };
        let mut end = self.parser.expect(Token::Colon)?.span.end;
        self.parser.expect(Token::Newline)?;
        self.parser.next_line()?;
        self.parser.expect(Token::IncreaseIndent)?;
        let mut body = Vec::new(self.arena);
        loop {
            let stmt = self.statement()?;
            body.push(stmt)?;
            self.parser.next_line()?;
            match self.parser.peek() {
                Some(Token::DecreaseIndent) => {
                    self.parser.skip()?;
                    break
                },
                Some(_) => {
                    continue
                }
                None => return Err(self.parser.unexpected(&"a statement, or decrease in indentation"))
            }
        }
        let span = Span { start, end };
        let body = &*body.into_slice();
        Ok(self.stmt(if is_async {
            StmtKind::FunctionDef {
                span, name,
                args: arg_declarations,
                body,
                decorator_list: decorators,
                returns: return_type,
                type_comment: None
            }
        } else {
            StmtKind::AsyncFunctionDef {
                span, name,
                args: arg_declarations,
                body,
                decorator_list: decorators,
                returns: return_type,
                type_comment: None
            }
        })?)
    }
    fn parse_class_def(&mut self, decorators: &[Expr<'a>]) -> Result<Stmt<'a>, ParseError> {
        todo!()
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