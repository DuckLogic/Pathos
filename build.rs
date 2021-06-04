use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{env, io};

fn run_python(module: &str, args: Vec<String>) -> Result<(), BuildError> {
    let give_error = |cause: Option<io::Error>| BuildError::FailedRunPython {
        module: module.into(), args: args.clone(), cause
    };
    match Command::new("python3")
        .args(&["-m", module])
        .args(&args)
        .stdout(Stdio::null())
        .spawn().and_then(|mut child| child.wait()) {
        Ok(code) => {
            if code.success() {
                Ok(())
            } else {
                Err(give_error(None))
            }
        },
        Err(cause) => {
            Err(give_error(Some(cause)))
        }
    }
}

pub fn main() -> Result<(), BuildError> {
    // Rerun if the python code changes
    println!("cargo:rerun-if-changed=pathos");
    let in_file = Path::new("src/Python.asdl");
    println!("cargo:rerun-if-changed={}", in_file.display());
    let out_file = PathBuf::from(
        env::var("OUT_DIR").unwrap()
    ).join("ast_gen.rs");
    run_python(
        "pathos.asdl.rust",
        vec![
            "-R".into(),
            out_file.to_string_lossy().into_owned(),
            in_file.to_string_lossy().into_owned(),
        ]
    )?;
    Ok(())
}

#[derive(Debug)]
pub enum BuildError {
    FailedRunPython {
        module: String,
        args: Vec<String>,
        cause: Option<io::Error>
    }
    
}