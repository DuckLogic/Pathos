#![feature(backtrace, never_type)]
use clap::{Clap, ArgGroup, ValueHint, AppSettings};
use pathos_python_parser::ParseMode;
use std::path::{PathBuf};
use anyhow::{Error, Context};
use std::ffi::OsString;
use tempfile::NamedTempFile;
use pathos::ast::{Span};
use pathos::alloc::Allocator;
use bumpalo::Bump;
use pathos_python_ast::constants::ConstantPool;
use pathos::ast::ident::SymbolTable;
use anyhow::{bail};
use std::io::{Read, Write};
use pathos_python_parser::lexer::{PythonLexer};
use pathos::errors::fancy::{FancyErrorContext, FancyErrorTarget};
use pathos::errors::tracker::LineTracker;
use ariadne::Source;
use pathos::errors::{SpannedError, ErrorSpan};
use std::fmt::{Debug, Display};
use std::backtrace::{Backtrace, BacktraceStatus};

/// Command line interface to the Pathos python parser
///
/// Includes a small set of tools that are useful
#[derive(Clap, Debug)]
#[clap(setting = AppSettings::InferSubcommands)]
#[clap(
    author, // Derive author from Cargo.toml
    version, // Derive version from Cargo.toml
    name = "pathos"
)]
enum Opt {
    /// A simple command to parse Python sources,
    /// then dump its AST
    ///
    /// By default, this dumps a JSON representation
    /// that should match the CPython AST.
    /// However, other representations should (eventually)
    /// be implemented.
    ///
    /// See the (unofficial) documentation of the official AST:
    /// https://greentreesnakes.readthedocs.io/en/latest/nodes.html
    DumpAst(DumpAstOptions),
    /// Tokenize a python source file, without doing any lexing
    ///
    /// Should (mostly) be equivalent to the builtin 'tokenize' command
    Tokenize(TokenizeOptions)
}

fn main() -> anyhow::Result<()> {
    let options: Opt = Opt::parse();
    match options {
        Opt::DumpAst(ref inner) => dump_ast(inner),
        Opt::Tokenize(ref inner) => tokenize(inner)
    }
}
fn tokenize(options: &TokenizeOptions) -> anyhow::Result<()> {
    let input = options.input.input()?;
    let text = input.read_input()?;
    let arena = Allocator::new(Bump::new());
    let symbols = SymbolTable::new(&arena);
    let mut lexer = PythonLexer::new(&arena, symbols, &text);
    let line_tracker = if !options.raw_spans {
        Some(LineTracker::from_text(&text))
    } else {
        None
    };
    let fancy_source: Source;
    let fancy_errors = if options.fancy_errors {
        let line_tracker = line_tracker.as_ref().ok_or_else(|| anyhow::anyhow!("The --fancy-errors flag isn't compatible with --raw-span"))?;
        fancy_source = Source::from(&text);
        Some(FancyErrorContext {
            tracker: line_tracker,
            current_file: &fancy_source,
        })
    } else {
        None
    };
    let display_span = |span: Span| {
        if let Some(ref tracker) = line_tracker {
            format!("{}", tracker.resolve_span(span))
        } else {
            format!("{}", span)
        }
    };
    let mut raw_text = String::new();
    let mut quoted_text = String::new();
    while let Some((span, token)) = {
        match lexer.try_next() {
            Ok(Some(tk)) => Some((lexer.current_span(), tk)),
            Err(err) => {
                if let Some(ref fancy_err_ctx) = fancy_errors {
                    options.error_opts.report_error(fancy_err_ctx, &err)?;
                } else if let ErrorSpan::Span(span) = err.span() {
                    return Err(err).with_context(|| format!("Error at {}", display_span(span)));
                } else {
                    return Err(err.into())
                }
            },
            Ok(None) => None
        }
    } {
        use std::fmt::Write;
        raw_text.clear();
        write!(&mut raw_text, "{}", token).unwrap();
        let num_single_quotes = raw_text.chars().filter(|&c| c == '\'').count();
        let num_double_quotes = raw_text.chars().filter(|&c| c == '"').count();
        let use_single_quotes = num_single_quotes <= num_double_quotes;
        quoted_text.clear();
        if use_single_quotes {
            quoted_text.push('\'');
        } else {
            quoted_text.push('"');
        }
        for c in raw_text.chars() {
            match (c, use_single_quotes) {
                ('\'', true) => {
                    quoted_text.push_str("\\'");
                },
                ('"', false) => {
                    quoted_text.push_str("\\\"");
                }
                ('"', true) | ('\'', false) |
                ('A'..='Z', _) | ('a'..='z', _) |
                ('0'..='9', _) | (' ', _) => {
                    quoted_text.push(c);
                }
                (other, _) => {
                    quoted_text.extend(other.escape_debug());
                }
            }
        }
        if use_single_quotes {
            quoted_text.push('\'');
        } else {
            quoted_text.push('"');
        }
        println!("{: <20}{}", display_span(span), quoted_text);
    }
    Ok(())
}
fn dump_ast(options: &DumpAstOptions) -> anyhow::Result<()> {
    let input = options.input.input()?;
    let text = input.read_input()?;
    let arena = Allocator::new(Bump::new());
    let mut pool = ConstantPool::new(&arena);
    let mut symbols = SymbolTable::new(&arena);
    let ast = match pathos_python_parser::parse_text(
        &arena,
        &text,
        options.parse_mode.clone().map_or_else(|| input.default_parse_mode(), ParseMode::from),
        &mut pool,
        &mut symbols
    ) {
        Ok(tree) => tree,
        Err(parse_error) => {
            let tracker = LineTracker::from_text(&text);
            let source = Source::from(&text);
            let ctx = FancyErrorContext {
                current_file: &source,
                tracker: &tracker
            };
            options.error_opts.report_error(&ctx, &parse_error)?;
        }
    };
    eprintln!("Pretty printed ast:");
    match options.output_format.unwrap_or_else(Default::default) {
        OutputFormat::Json => {
            let out = std::io::stdout();
            let mut out = out.lock();
            ::serde_json::to_writer_pretty(
                &mut out,
                &ast
            )?;
            out.write_all(b"\n")?;
            out.flush()?;
            drop(out);
        },
        OutputFormat::Debug => {
            println!("{:#?}", ast)
        }
    }
    if options.verbose {
        eprintln!("Allocated {} bytes", arena.allocated_bytes());
    }
    Ok(())
}

#[derive(Clap, Debug, Clone, Copy)]
pub enum OutputFormat {
    Json,
    Debug
}
impl Default for OutputFormat {
    fn default() -> OutputFormat {
        OutputFormat::Json
    }
}

#[derive(Clap, Debug)]
struct TokenizeOptions {
    /// Output raw spans, without tracking any line information
    #[clap(long)]
    raw_spans: bool,
    /// Use fancy errors
    ///
    /// Requires that raw-spans is false
    #[clap(long = "fancy")]
    fancy_errors: bool,
    #[clap(flatten)]
    error_opts: ErrorOptions,
    #[clap(flatten)]
    input: InputOptions
}

#[derive(Clap, Debug)]
struct ErrorOptions {
    /// Print the internal backtrace of errors
    ///
    /// This should only be used during debugging of the parser's internals.
    #[clap(long, alias = "backtrace")]
    print_error_backtrace: bool,
    /// Print the internal causes of the error
    #[clap(long)]
    print_internal_causes: bool
}
impl ErrorOptions {
    fn print_backtrace(&self, bt: Option<&Backtrace>) {
        if let Some(bt) = bt {
            match bt.status() {
                BacktraceStatus::Unsupported => {
                    eprintln!("Backtrace: Unsupported");
                },
                BacktraceStatus::Disabled => {
                    eprintln!("Backtrace: Disabled");
                    eprintln!("  Consider setting $RUST_BACKTRACE=1");
                },
                BacktraceStatus::Captured => {
                    eprintln!("Backtrace:");
                    eprintln!("{}", bt)
                },
                _ => {
                    eprintln!("Backtrace: Unknown status {:?}", bt.status())
                }
            }
        } else {
            eprintln!("Backtrace: Missing");
        }
    }
    fn report_error<E>(&self, ctx: &FancyErrorContext, error: &E) -> anyhow::Result<!>
        where E: FancyErrorTarget + std::error::Error + 'static {
        struct SourceRef<'a>(&'a Source);
        impl<'a> ariadne::Cache<()> for SourceRef<'a> {
            fn fetch(&mut self, _id: &()) -> Result<&Source, Box<dyn Debug + '_>> {
                Ok(self.0)
            }

            fn display<'a2>(&self, _id: &'a2 ()) -> Option<Box<dyn Display + 'a2>> {
                None
            }
        }
        ctx.report_error(error).eprint(SourceRef(ctx.current_file))?;
        if self.print_error_backtrace {
            self.print_backtrace(error.backtrace());
        }
        if self.print_internal_causes {
            let mut source = error.source();
            while let Some(e) = source {
                eprintln!("Cause: {:?}", e);
                if self.print_error_backtrace {
                    self.print_backtrace(e.backtrace());
                }
                source = e.source();
            }
        }
        std::process::exit(1);
    }
}

#[derive(Clap, Debug)]
struct DumpAstOptions {
    /// Give verbose output
    #[clap(long, short = 'v')]
    verbose: bool,
    #[clap(flatten)]
    error_opts: ErrorOptions,
    /// The format to output the parsed AST in
    ///
    /// By default, this is inferred
    #[clap(long, arg_enum)]
    output_format: Option<OutputFormat>,
    /// The mode to parse the input in
    #[clap(long, arg_enum)]
    parse_mode: Option<ParseModeOpt>,
    #[clap(flatten)]
    input: InputOptions
}
#[derive(Clap, Debug)]
#[clap(group = ArgGroup::new("input").required(true))]
pub struct InputOptions {
    /// The input source file
    #[clap(
        long, short = 'c',
        aliases = &["file"],
        group = "input",
        parse(from_os_str),
        value_hint = ValueHint::FilePath
    )]
    input_file: Option<PathBuf>,
    /// Explicitly given input text
    #[clap(long = "text", group = "input")]
    input_text: Option<String>,
    /// Read the input from stdin
    #[clap(long = "stdin", group = "input")]
    read_stdin: bool,
    /// Open the input in an editor
    #[clap(long = "edit", group = "input")]
    open_editor: bool,
}
impl InputOptions {
    fn input(&self) -> Result<Input, Error> {
        if let Some(ref f) = self.input_file {
            Ok(Input::File(f.clone()))
        } else if let Some(ref t) = self.input_text {
            Ok(Input::Text(t.clone()))
        } else if self.read_stdin {
            Ok(Input::Stdin)
        } else if self.open_editor {
            Ok(Input::Editor)
        } else {
            bail!("No input given")
        }
    }
}
enum Input {
    File(PathBuf),
    Text(String),
    Stdin,
    Editor
}
impl Input {
    fn default_parse_mode(&self) -> ParseMode {
        match *self {
            Input::File(_) => ParseMode::Module,
            Input::Text(_) => ParseMode::Expression,
            Input::Stdin => ParseMode::Module,
            Input::Editor => ParseMode::Module
        }
    }
    fn read_input(&self) -> Result<String, Error> {
        match *self {
            Input::File(ref path) => {
                let mut res = String::new();
                std::fs::File::open(path).and_then(|mut f| f.read_to_string(&mut res))
                    .with_context(|| format!("Failed to read file: {}", path.display()))?;
                Ok(res)
            }
            Input::Text(ref t) => Ok(t.clone()),
            Input::Stdin => {
                let mut res = String::new();
                let stdin = std::io::stdin();
                let mut stdin = stdin.lock();
                stdin.read_to_string(&mut res)
                    .with_context(|| "Unable to read stdin".to_string())?;
                drop(stdin);
                Ok(res)
            },
            Input::Editor => {
                let editor = std::env::var_os("EDITOR")
                    .unwrap_or_else(|| OsString::from("vi"));
                let mut temp = NamedTempFile::new()?;
                use std::process::{Command};
                let mut proc = Command::new(&editor)
                    .arg(temp.path().as_os_str())
                    .spawn()
                    .with_context(|| format!("Unable to spawn editor '{}'", editor.to_string_lossy()))?;
                proc.wait().map_err(anyhow::Error::from)
                    .and_then(|status| if !status.success() {
                        bail!("Failure w/ status {}", status)
                    } else { Ok(()) })
                    .with_context(|| format!("Failed to run editor '{}'", editor.to_string_lossy()))?;
                let mut res = String::new();
                temp.read_to_string(&mut res)?;
                Ok(res)
            }
        }
    }
}
#[derive(Clap, Debug, Clone)]
enum ParseModeOpt {
    #[clap(alias = "expr", alias = "eval")]
    Expression,
    #[clap(alias = "mod", alias = "exec")]
    Module,
}
impl From<ParseModeOpt> for ParseMode {
    fn from(opt: ParseModeOpt) -> Self {
       match opt {
           ParseModeOpt::Expression => ParseMode::Expression,
           ParseModeOpt::Module => ParseMode::Module,
       }
    }
}
