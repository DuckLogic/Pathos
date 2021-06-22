use errors::ParseError;

use crate::alloc::Allocator;
use crate::ast::constants::ConstantPool;
use crate::ast::ident::{Ident, Symbol};
use crate::ast::tree::{ExprContext, Arguments, ArgumentStyle, Arg, PythonAst};
use crate::lexer::Token;

pub use self::expr::ExprPrec;
use self::parser::{IParser, Parser, SpannedToken};
use crate::ast::{Span, Spanned};
use crate::parse::errors::ParseErrorKind;
use crate::ParseMode;

pub mod errors;
pub mod parser;
mod expr;
mod stmt;


#[derive(Debug)]
pub struct PythonParser<'src, 'a, 'p> {
    pub arena: &'a Allocator,
    pub parser: Parser<'src, 'a>,
    expression_context: ExprContext,
    pub pool: &'p mut ConstantPool<'a>,
}
impl<'src, 'a, 'p> PythonParser<'src, 'a, 'p> {
    pub fn new(arena: &'a Allocator, parser: Parser<'src, 'a>, pool: &'p mut ConstantPool<'a>) -> Self {
        PythonParser {
            arena, parser, pool,
            expression_context: Default::default(),
        }
    }
    pub fn parse_top_level(&mut self, mode: ParseMode) -> Result<PythonAst<'a>, ParseError> {
        use crate::alloc::Vec;
        match mode {
            ParseMode::Expression => Ok(PythonAst::Expression {
                body: self.expression()?
            }),
            ParseMode::Module => {
                let mut stmts = Vec::new(self.arena);
                while !self.parser.is_end_of_file() {
                    let stmt = self.statement()?;
                    stmts.push(stmt)?;
                }
                Ok(PythonAst::Module {
                    body: stmts.into_slice(),
                    type_ignores: &[]
                })
            }
        }
    }
    #[inline]
    pub fn parse_ident(&mut self) -> Result<Ident<'a>, ParseError> {
        let span = self.parser.current_span();
        let symbol = self.parser.expect_map(&"an identifier", |token| match token.kind {
            Token::Ident(ident) => Some(ident),
            _ => None
        })?;
        Ok(Ident {
            symbol,
            span
        })
    }
    /// Parses a list of argument declarations
    ///
    /// This corresponds to the "parameter_list" in the grammar of function definitions:
    /// <https://docs.python.org/3.10/reference/compound_stmts.html#function-definitions>
    pub fn parse_argument_declarations(&mut self, ending_token: Token<'a>, opts: ArgumentParseOptions) -> Result<&'a Arguments<'a>, ParseError>{
        use crate::alloc::Vec;
        let start = self.parser.current_span().start;
        /*
         * NOTE: It is possible we are empty, in which case our span doesn't even include
         * the span of the current token.
         */
        let mut end = start;
        let mut positional_args = Vec::new(self.arena);
        let mut keyword_args = Vec::new(self.arena);
        let mut keyword_vararg = None;
        let mut vararg = None;
        /*
         * This is essentially our state, or rather it's what we expect to see next.
         * We start out parsing positional arguments. If we see '/', we switch to parsing
         * positional-only arguments. If we see '*vargs' or '*,',
         * we then switch to parsing keyword-only arguments.
         * NOTE: That `ArgumentStyle::Vararg` and `ArgumentStyle::KeywordVararg` should be impossible,
         * since there are only one of each of those items, it doesn't make sense to keep that as a long-term state.
         * Furthermore, If we see a keyword vararg '**kwargs', then we are completely done parsing,
         * and should return.
         */
        let mut current_style = ArgumentStyle::Positional;
        /*
         * This 'mode' of parsing indicates whether or not
         * default arguments are optional, forbidden, or required.
         */
        let mut current_default_handling = DefaultArgumentHandling::Standard;
        /// Give a good error message depending on our current state (or rather, current ArgumentStyle)
        #[cold]
        fn unexpected_item_msg<'a>(current_style: ArgumentStyle, ending_token: &Token<'a>) -> Box<str> {
            let primary_msg = match current_style {
                ArgumentStyle::Positional => "an arg, '/', '*', '**'",
                ArgumentStyle::PositionalOnly => "an arg, '*', '**'",
                ArgumentStyle::Keyword => {
                    /*
                     * NOTE: `*items,` functions  equivalently to a keyword-only specifier `*`
                     */
                    "a keyword arg, or '**'"
                },
                ArgumentStyle::KeywordVararg | ArgumentStyle::Vararg => unreachable!()
            };
            format!("{}, or {}", primary_msg, ending_token).into_boxed_str()
        }
        'argParsing: loop {
            match self.parser.peek() {
                Some(Token::Slash) if current_style == ArgumentStyle::Positional => {
                    // Switch to positional only args
                    self.parser.skip()?;
                    current_style = ArgumentStyle::PositionalOnly;
                    self.parser.expect(Token::Comma)?;
                    /*
                     * TODO: Somehow indicate that we *require* more args after this
                     * In other words, 'def sneaky(a, b, c, \,)'
                     * We need to fix this for the keyword-only specifier too
                     */
                    continue 'argParsing;
                },
                Some(Token::Star) => {
                    match current_style {
                        ArgumentStyle::Positional | ArgumentStyle::PositionalOnly => {
                            /*
                             * From now on, we only accept keyword arguments.
                             * There are two possible cases:
                             * 1. This star begins a vararg-declaration `*items`.
                             * 2. This star is a keyword-only specifier '*, ...'
                             * Either way
                             */
                            current_style = ArgumentStyle::Keyword;
                            /*
                             * Switch to allow non-default args once again
                             * This is needed in case we were previously in the 'require defaults' mode.
                             */
                            current_default_handling = DefaultArgumentHandling::Standard
                        }
                        ArgumentStyle::Keyword => {
                            let is_vararg_declaration = matches!(
                                /*
                                 * TODO: An error here would generate an unhelpful message.
                                 * However, swallowing it would be even worse.
                                 */
                                self.parser.look_ahead(1)?,
                                Some(SpannedToken { kind: Token::Ident(_), .. })
                            );
                            return Err(ParseError::builder(self.parser.current_span(), ParseErrorKind::UnexpectedToken)
                                .expected("More keyword argument names")
                                .actual(if is_vararg_declaration {
                                    "A vararg declaration"
                                } else {
                                    "A keyword-only specifier '*'"
                                })
                                .build())
                        },
                        ArgumentStyle::KeywordVararg | ArgumentStyle::Vararg => unreachable!()
                    }
                    self.parser.expect(Token::Star)?;
                    match self.parser.peek() {
                        Some(Token::Ident(_)) => {
                            assert_eq!(vararg, None, "Already parsed a vararg");
                            vararg = Some(&*self.arena.alloc(self.parse_single_argument_declaration(
                                &DefaultArgumentHandling::ForbidDefaults {
                                    reason: "for vararg parameter"
                                }, opts,
                                ArgumentStyle::Vararg
                            )?)?);
                            end = vararg.unwrap().span().end;
                            // Fallthrough to comma/closing
                        },
                        Some(Token::Comma) => {
                            /*
                             * We're a keyword-only specifier.
                             * Consume the comma and continue parsing
                             */
                            end = self.parser.current_span().end;
                            self.parser.skip()?;
                            continue 'argParsing;
                        },
                        _ => {
                            /*
                             * NOTE: A closing after a keyword-only specifier is invalid.
                             * We *must* have a comma. In other words,
                             * "def bad(a, b, c, *)" is invalid.
                             */
                            return Err(self.parser.unexpected(&"a comma or a vararg-name"))
                        }
                    }
                    // Fallthrough to comma/closing
                },
                Some(Token::DoubleStar) => {
                    self.parser.skip()?;
                    keyword_vararg = Some(&*self.arena.alloc(self.parse_single_argument_declaration(
                        &DefaultArgumentHandling::ForbidDefaults {
                            reason: "for keyword vararg parameter"
                        },
                        opts, ArgumentStyle::KeywordVararg
                    )?)?);
                    end = keyword_vararg.unwrap().span().end;
                    break 'argParsing;
                }
                Some(other) if other == ending_token => {
                    break 'argParsing;
                }
                _ => {
                    let arg = match self.parse_single_argument_declaration(&current_default_handling, opts, current_style) {
                        Ok(arg) => arg,
                        Err(e) => {
                            return Err(e.with_expected_msg(unexpected_item_msg(current_style, &ending_token)))
                        }
                    };
                    end = arg.span.end;
                    match current_style {
                        ArgumentStyle::Positional | ArgumentStyle::PositionalOnly => {
                            if arg.default_value.is_some() {
                                current_default_handling = DefaultArgumentHandling::RequireDefaults {
                                    first_arg_with_default: arg.name.symbol
                                }
                            }
                            positional_args.push(arg)?;
                        },
                        ArgumentStyle::Keyword => {
                            keyword_args.push(arg)?;
                        }
                        ArgumentStyle::Vararg | ArgumentStyle::KeywordVararg => unreachable!()
                    }
                }
            }
            // Fallthrough: Either parse a comma or ending
            match self.parser.peek() {
                Some(Token::Comma) => {
                    end = self.parser.current_span().end;
                    self.parser.skip()?;
                    continue 'argParsing;
                },
                Some(other) if other == ending_token => {
                    break 'argParsing;
                },
                _ => {
                    return Err(self.parser.unexpected(&format_args!("a comma or '{}'", ending_token)));
                }
            }
        }
        Ok(&*self.arena.alloc(Arguments {
            positional_args: positional_args.into_slice(),
            keyword_args: keyword_args.into_slice(),
            keyword_vararg, vararg, span: Span { start, end }
        })?)
    }
    /// Parse a single argument declaration
    ///
    /// This accepts an [ArgumentParseMode] to control the specifics of parsing.
    ///
    /// In the grammar of function
    fn parse_single_argument_declaration(&mut self, default_mode: &DefaultArgumentHandling, opts: ArgumentParseOptions, style: ArgumentStyle) -> Result<Arg<'a>, ParseError> {
        let name = self.parse_ident().map_err(|parser| parser.with_expected_msg("An argument declaration"))?;
        let start = name.span.start;
        let mut end = name.span.end;
        let mut default_value = None;
        let mut type_annotation = None;
        match self.parser.peek() {
            Some(Token::Colon) if opts.allow_type_annotations => {
                self.parser.skip()?;
                type_annotation = Some(self.expression()?);
                end = type_annotation.unwrap().span().end;
            },
            Some(Token::Equals) => {
                // Fallthrough to default_value parsing code
            },
            _ => {
                // Anything else means we are done
                return Ok(Arg {
                    span: Span { start, end },
                    name, default_value: None,
                    type_comment: None,
                    annotation: None, style
                })
            }
        }
        if let Some(Token::Equals) = self.parser.peek() {
            if let DefaultArgumentHandling::ForbidDefaults { reason } = default_mode {
                return Err(self.parser.unexpected(&format_args!("Forbidden default argument {}", reason)));
            } else {
                self.parser.expect(Token::Equals)?;
                default_value = Some(self.expression()?);
                end = default_value.unwrap().span().end;
            }
        } else if let DefaultArgumentHandling::RequireDefaults { first_arg_with_default } = default_mode {
            return Err(self.parser.unexpected(&format_args!(
                "Default argument required because {} had a default arg",
                first_arg_with_default
            )));
        }
        Ok(Arg {
            name, default_value, style,
            span: Span { start, end }, annotation: type_annotation,
            type_comment: None // TODO: Type comments
        })
    }
}
impl<'src, 'a, 'p> IParser<'src, 'a> for PythonParser<'src, 'a, 'p> {
    #[inline]
    fn as_mut_parser(&mut self) -> &mut Parser<'src, 'a> {
        &mut self.parser
    }
    #[inline]
    fn as_parser(&self) -> &Parser<'src, 'a> {
        &self.parser
    }
}
impl Default for ExprContext {
    #[inline]
    fn default() -> Self {
        ExprContext::Load
    }
}
#[derive(Copy, Clone, Debug)]
pub struct ArgumentParseOptions {
    pub allow_type_annotations: bool
}
impl Default for ArgumentParseOptions {
    #[inline]
    fn default() -> Self {
        ArgumentParseOptions {
            allow_type_annotations: true
        }
    }
}
#[derive(Clone, educe::Educe)]
#[educe(Debug)]
enum DefaultArgumentHandling<'a> {
    Standard,
    /// Forbid a default value for the argument
    ForbidDefaults {
        reason: &'static str,
    },
    /// Require a default value for this argument,
    /// because another value already had a default
    RequireDefaults {
        first_arg_with_default: Symbol<'a>
    }
}
impl Default for DefaultArgumentHandling<'_> {
    #[inline]
    fn default() -> Self {
        DefaultArgumentHandling::Standard
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::ast::{AstBuilder, Constant};
    use std::cell::RefCell;
    use crate::ast::ident::SymbolTable;
    use crate::ast::tree::{Expr, ExprKind, Operator};
    use crate::ast::constants::{StringStyle, StringLiteral};
    use crate::lexer::PythonLexer;
    use bumpalo::Bump;

    macro_rules! vec {
        ($ctx:expr) => (vec![$ctx,]);
        ($ctx:expr, $($elements:expr),*) => {
            &*crate::vec![in $ctx.arena; $($elements),*].unwrap().into_slice()
        };
    }

    pub struct TestContext<'a, 'p> {
        pub arena: &'a Allocator,
        pub builder: AstBuilder<'a>,
        pub pool: RefCell<&'p mut ConstantPool<'a>>,
        pub symbol_table: RefCell<&'p mut SymbolTable<'a>>
    }
    impl<'a, 'p> TestContext<'a, 'p> {
        pub fn int(&self, i: i64) -> Expr<'a> {
            let mut pool = self.pool.borrow_mut();
            let val = pool.int(DUMMY, i).unwrap();
            self.constant(val)
        }
        pub fn bool(&self, b: bool) -> Expr<'a> {
            let mut pool = self.pool.borrow_mut();
            let val = pool.bool(DUMMY, b).unwrap();
            self.constant(val)
        }
        pub fn str_lit(&self, s: &'static str) -> Expr<'a> {
            let mut pool = self.pool.borrow_mut();
            let val = pool.string(DUMMY, StringLiteral {
                value: s,
                style: StringStyle::double_quote()
            }).unwrap();
            self.constant(val)
        }
        pub fn ident(&self, s: &'static str) -> Ident<'a> {
            let mut symbol_table = self.symbol_table.borrow_mut();
            Ident {
                symbol: symbol_table.alloc(s).unwrap(),
                span: DUMMY
            }
        }
        pub fn attr_ref(&self, base: Expr<'a>, name: &'static str) -> Expr<'a> {
            self.builder.expr(ExprKind::Attribute {
                ctx: ExprContext::Load,
                value: base,
                attr: self.ident(name),
                span: DUMMY
            }).unwrap()
        }
        pub fn expr(&self, e: ExprKind<'a>) -> Expr<'a> {
            self.builder.expr(e).unwrap()
        }
        pub fn bin_op(&self, left: Expr<'a>, op: Operator, right: Expr<'a>) -> Expr<'a> {
            self.expr(ExprKind::BinOp { left, op, right, span: DUMMY })
        }
        pub fn name(&self, s: &'static str) -> Expr<'a> {
            self.arena.alloc(ExprKind::Name {
                span: DUMMY, id: self.ident(s),
                ctx: ExprContext::Load
            }).unwrap()
        }
        pub fn constant(&self, value: Constant<'a>) -> Expr<'a> {
            self.arena.alloc(ExprKind::Constant {
                span: DUMMY,
                kind: None,
                value
            }).unwrap()
        }
        pub fn arg(&self, name: &'static str, type_annotation: Option<Expr<'a>>, default_value: Option<Expr<'a>>, style: ArgumentStyle) -> Arg<'a> {
            Arg {
                name: self.ident(name),
                span: DUMMY,
                type_comment: None,
                annotation: type_annotation,
                style,
                default_value
            }
        }
        pub fn keyword_arg(&self, name: &'static str, type_annotation: Option<Expr<'a>>, default_value: Option<Expr<'a>>) -> Arg<'a> {
            self.arg(name, type_annotation, default_value, ArgumentStyle::Keyword)
        }
        pub fn simple_arg(&self, name: &'static str, type_annotation: Option<&'static str>) -> Arg<'a> {
            self.arg(name, type_annotation.map(|tp| self.name(tp)), None, ArgumentStyle::Positional)
        }
        pub fn arg_declarations(&self, args: &[Arg<'a>]) -> Arguments<'a> {
            use crate::alloc::Vec;
            let mut positional_args = Vec::new(self.arena);
            let mut keyword_args = Vec::new(self.arena);
            let mut vararg = None;
            let mut keyword_vararg = None;
            for arg in args {
                match arg.style {
                    ArgumentStyle::Positional | ArgumentStyle::PositionalOnly => {
                        assert!(keyword_args.is_empty());
                        assert_eq!(vararg, None);
                        assert_eq!(keyword_vararg, None);
                        positional_args.push(arg.clone()).unwrap();
                    }
                    ArgumentStyle::Vararg => {
                        assert!(keyword_args.is_empty());
                        assert_eq!(vararg, None);
                        assert_eq!(keyword_vararg, None);
                        vararg = Some(&*self.arena.alloc(arg.clone()).unwrap());
                    }
                    ArgumentStyle::Keyword => {
                        assert_eq!(keyword_vararg, None);
                        keyword_args.push(arg.clone()).unwrap();
                    }
                    ArgumentStyle::KeywordVararg => {
                        assert_eq!(keyword_vararg, None);
                        keyword_vararg = Some(&*self.arena.alloc(arg.clone()).unwrap());
                    }
                }
            }
            Arguments {
                positional_args: positional_args.into_slice(), keyword_vararg,
                keyword_args: keyword_args.into_slice(), vararg,
                span: DUMMY
            }
        }
    }
    fn test_arg_declaration(s: &str, expected: &mut dyn for<'a, 'p> FnMut(&mut TestContext<'a, 'p>) -> Arguments<'a>) {
        let arena = Allocator::new(Bump::new());
        let mut pool = ConstantPool::new(&arena);
        let mut symbol_table = Some(SymbolTable::new(&arena));
        let lexer = PythonLexer::new(
            &arena,
            symbol_table.take().unwrap(),
            s
        );
        let raw_parser = Parser::new(&arena, lexer).unwrap();
        let mut parser = PythonParser::new(&arena, raw_parser, &mut pool);
        parser.parser.expect(Token::OpenParen).unwrap();
        let res = parser.parse_argument_declarations(Token::CloseParen, ArgumentParseOptions::default()).unwrap();
        parser.parser.expect(Token::CloseParen).unwrap();
        parser.parser.expect_end_of_input().unwrap();
        assert!(symbol_table.is_none());
        symbol_table = Some(parser.parser.into_lexer().into_symbol_table());
        let mut ctx = TestContext {
            arena: &arena,
            builder: AstBuilder { arena: &arena },
            symbol_table: RefCell::new(symbol_table.as_mut().unwrap()),
            pool: RefCell::new(&mut pool)
        };
        assert_eq!(*res, expected(&mut ctx));
    }
    pub const DUMMY: Span = Span::dummy();

    #[test]
    fn arg_declarations() {
        // Simple arg declarations
        test_arg_declaration("()", &mut |ctx| {
            ctx.arg_declarations(&[])
        });
        test_arg_declaration("(a, b, c)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", None),
                ctx.simple_arg("b", None),
                ctx.simple_arg("c", None)
            ))
        });
        test_arg_declaration("(arg1,)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("arg1", None)
            ))
        });
        test_arg_declaration("(a: int, b, c: str)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("int")),
                ctx.simple_arg("b", None),
                ctx.simple_arg("c", Some("str"))
            ))
        });
        // Add in some default argument values
        test_arg_declaration(r#"(a=1, b: bool = True, c: str = "cool", d=1)"#, &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.arg("a", None, Some(ctx.int(1)), ArgumentStyle::Positional),
                ctx.arg("b", Some(ctx.name("bool")), Some(ctx.bool(true)), ArgumentStyle::Positional),
                ctx.arg("c", Some(ctx.name("str")), Some(ctx.str_lit("cool")), ArgumentStyle::Positional),
                ctx.arg("d", None, Some(ctx.int(1)), ArgumentStyle::Positional)
            ))
        });
        // Try some positional only args
        test_arg_declaration(r#"(a: bool, b: ClassType, /, c: bool)"#, &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("bool")),
                ctx.simple_arg("b", Some("ClassType")),
                ctx.arg("c", Some(ctx.name("bool")), None, ArgumentStyle::PositionalOnly)
            ))
        });
        // Add in varargs
        test_arg_declaration(r#"(a: bool, b: ClassType, /, c: bool, *items: int)"#, &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("bool")),
                ctx.simple_arg("b", Some("ClassType")),
                ctx.arg("c", Some(ctx.name("bool")), None, ArgumentStyle::PositionalOnly),
                ctx.arg("items", Some(ctx.name("int")), None, ArgumentStyle::Vararg)
            ))
        });
        // Keyword-only specifiers
        test_arg_declaration(r#"(a: bool, b: ClassType, /, c: bool = True, *, d: int = 3)"#, &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("bool")),
                ctx.simple_arg("b", Some("ClassType")),
                ctx.arg("c", Some(ctx.name("bool")), Some(ctx.bool(true)), ArgumentStyle::PositionalOnly),
                ctx.keyword_arg("d", Some(ctx.name("int")), Some(ctx.int(3)))
            ))
        });
        // A varargs functions just like a keyword-only specifier
        test_arg_declaration("(a, b, *items, e)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", None),
                ctx.simple_arg("b", None),
                ctx.arg("items", None, None, ArgumentStyle::Vararg),
                ctx.keyword_arg("e", None, None)
            ))
        });
        test_arg_declaration(r#"(a: bool, b: ClassType, /, c: bool = True, *items, d: int = 3, e)"#, &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("bool")),
                ctx.simple_arg("b", Some("ClassType")),
                ctx.arg("c", Some(ctx.name("bool")), Some(ctx.bool(true)), ArgumentStyle::PositionalOnly),
                ctx.arg("items", None, None, ArgumentStyle::Vararg),
                ctx.keyword_arg("d", Some(ctx.name("int")), Some(ctx.int(3))),
                ctx.keyword_arg("e", None, None)
            ))
        });
        // Try keyword-only specifier without positional only
        test_arg_declaration("(a: bool, b: ClassType, *, d: int = 3, e)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("bool")),
                ctx.simple_arg("b", Some("ClassType")),
                ctx.keyword_arg("d", Some(ctx.name("int")), Some(ctx.int(3))),
                ctx.keyword_arg("e", None, None)
            ))
        });
        // Try the `**kwargs` collector
        test_arg_declaration("(a, b, **kwargs)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", None),
                ctx.simple_arg("b", None),
                ctx.arg("kwargs", None, None, ArgumentStyle::KeywordVararg)
            ))
        });
        // Try the `**kwargs` collector + `*items` collector
        test_arg_declaration("(a, b, *items, **kwargs)", &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", None),
                ctx.simple_arg("b", None),
                ctx.arg("items", None, None, ArgumentStyle::Vararg),
                ctx.arg("kwargs", None, None, ArgumentStyle::KeywordVararg)
            ))
        });
        // Try all possible features, stitched together
        test_arg_declaration(r#"(a: bool, b: ClassType, /, c: bool = True, *items, d: int = 3, e, **kwargs)"#, &mut |ctx| {
            ctx.arg_declarations(vec!(
                ctx,
                ctx.simple_arg("a", Some("bool")),
                ctx.simple_arg("b", Some("ClassType")),
                ctx.arg("c", Some(ctx.name("bool")), Some(ctx.bool(true)), ArgumentStyle::PositionalOnly),
                ctx.arg("items", None, None, ArgumentStyle::Vararg),
                ctx.keyword_arg("d", Some(ctx.name("int")), Some(ctx.int(3))),
                ctx.keyword_arg("e", None, None),
                ctx.arg("kwargs", None, None, ArgumentStyle::KeywordVararg)
            ))
        });
    }
}