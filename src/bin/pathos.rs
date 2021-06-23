use clap::{Clap, ArgGroup, ValueHint, AppSettings};
use pathos_python_parser::ParseMode;
use std::path::{PathBuf};
use anyhow::{Error, Context};
use std::ffi::OsString;
use tempfile::NamedTempFile;
use pathos_python_parser::ast::Allocator;
use bumpalo::Bump;
use pathos_python_parser::ast::constants::ConstantPool;
use pathos_python_parser::ast::ident::SymbolTable;
use anyhow::{bail};
use std::io::{Read, Write};

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
    DumpAst(DumpAstOptions)
}

fn main() -> anyhow::Result<()> {
    let options: Opt = Opt::parse();
    match options {
        Opt::DumpAst(ref inner) => dump_ast(inner)
    }
}
fn dump_ast(options: &DumpAstOptions) -> anyhow::Result<()> {
    let input = options.input.input()?;
    let text = input.read_input()?;
    let arena = Allocator::new(Bump::new());
    let mut pool = ConstantPool::new(&arena);
    let mut symbols = SymbolTable::new(&arena);
    let ast = pathos_python_parser::parse_text(
        &arena,
        &text,
        options.parse_mode.clone().map_or_else(|| input.default_parse_mode(), ParseMode::from),
        &mut pool,
        &mut symbols
    ).with_context(|| format!("Failed to parse input"))?;
    eprintln!("Pretty printed ast:");
    match options.output_format.unwrap_or_else(Default::default) {
        OutputFormat::Json => {
            let out = std::io::stdout();
            let mut out = out.lock();
            ::serde_json::to_writer_pretty(
                &mut out,
                &ast
            )?;
            out.write(b"\n")?;
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
struct DumpAstOptions {
    /// Give verbose output
    #[clap(long, short = 'v')]
    verbose: bool,
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
                    .with_context(|| format!("Unable to read stdin"))?;
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
