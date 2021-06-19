use crate::parse::PythonParser;
use crate::ast::tree::{Stmt, StmtKind, ExprKind, Alias};
use crate::lexer::Token;
use crate::ParseError;
use crate::ast::{Spanned, Span};
use crate::parse::parser::{IParser, ParseSeperated};

impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    pub fn statement(&mut self) -> Result<Stmt<'a>, ParseError> {
        let stmt = match self.parser.peek() {
            Ok(Token::Assert) => self.parse_assertion()?,
            Ok(Token::Pass) => {
                let start = self.parser.expect(Token::Assert)?.span.start;
                let condition = self.expression()?;
                let message = if let Some(Token::Comma) = self.parser.peek() {
                    self.parser.skip()?;
                    Some(self.expression()?)
                } else { None };
                let end = message.as_ref().unwrap_or(&condition).span().end;
                self.stmt(StmtKind::Assert { span: Span { start, end }, test: condition, msg: message })
            },
            Ok(Token::Del) => {
                let start_span = self.parser.expect(Token::Del)?.span.start;
                let targets = self.parse_target_list(
                    &mut |parser| parser.parser.is_newline(),
                    "newline"
                )?;
                self.stmt(StmtKind::Delete {
                    span: Span { start: start_span, end: targets.span().unwrap().start },
                    targets: targets.into_slice()?
                })?
            },
            Ok(Token::Return) => {
                let start_span = self.parser.expect(Token::Return)?.span;
                let exprs  = self.parse_expression_list(
                    true, |parser| parser.is_newline(),
                    "newline"
                )?;
                self.stmt(StmtKind::Return {
                    span: Span { start: start_span.start, end: match exprs.span() {
                        Some(span) => span.end,
                        None => start_span.end
                    } },
                    value: exprs.with_implicit_tuple()?
                })
            }
            Ok(Token::Yield) => {
                let expr = self.parse_yield_expr(|parser| parser.is_newline())?;
                self.stmt(StmtKind::Expr {
                    value: expr, span: expr.span()
                })
            },
            Ok(Token::Raise) => {
                let mut span = self.parser.expect(Token::Raise)?.span;
                let expr;
                let from;
                if !self.parser.is_newline() {
                    expr = Some(self.expression()?);
                    span.end = expr.as_ref().unwrap().span().end;
                    if let Some(Token::From) = self.parser.peek() {
                        from = Some(self.expression()?)
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
            Ok(Token::Break) => {
                let span = self.parser.expect(Token::Break)?.span;
                self.arena.alloc(StmtKind::Break { span })?
            },
            Ok(Token::Continue) => {
                let span = self.parser.expect(Token::Continue)?.span;
                self.arena.alloc(StmtKind::Continue { span })?
            },
            Ok(Token::Import) => {
                let span = self.parser.expect(Token::Continue)?.span;
                self.arena.alloc(StmtKind::Import { span })?
            }
            _ => return Err(self.parser.unexpected(&"a statement"))
        };
    }
    #[inline]
    fn parse_import_alias(&mut self) -> Result<Alias<'a>, ParseError> {
        let mut iter = ParseSeperated::new(
            self, |parser| parser.parse_ident(),
            Token::Period, |parser| !matches!(parser.peek(), Token::Ident(_) | Token::As),
            false
        );
        while let Some(item) = iter.next().transpose()? {
            todo!()
        }
        let renamed = if let Some(Token::As) = self.parser.peek() {
            self.parser.skip()?;
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(todo!())
    }
    #[inline]
    fn stmt(&mut self, stmt: StmtKind) -> Result<Stmt<'a>, ParseError> {
        Ok(&*self.arena.alloc(stmt)?)
    }
    fn parse_assertion(&mut self) -> Result<Stmt<'a>, ParseError> {

    }

}