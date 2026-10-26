use anyhow::{bail, Result};
use ort::{
    execution_providers::DirectMLExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use std::path::{Path, PathBuf};
pub fn find_asset(relative: &str) -> PathBuf {
    for start in [
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf)),
        std::env::current_dir().ok(),
    ]
    .into_iter()
    .flatten()
    {
        for parent in start.ancestors() {
            let candidate = parent.join(relative);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(relative)
}

pub fn init_runtime() -> Result<()> {
    let dll = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| find_asset("onnxruntime.dll"));
    if !dll.is_file() {
        bail!("ONNX Runtime not found. Run package.ps1 to place the native DLLs beside the executable, or set ORT_DYLIB_PATH.");
    }
    ort::init_from(dll.to_string_lossy()).commit()?;
    Ok(())
}

pub fn create_session(model: &Path, gpu: bool, adapter: i32) -> Result<(Session, String)> {
    if gpu {
        let attempt = (|| -> Result<Session> {
            Ok(Session::builder()?
                .with_memory_pattern(false)?
                .with_parallel_execution(false)?
                .with_execution_providers([DirectMLExecutionProvider::default()
                    .with_device_id(adapter)
                    .build()
                    .error_on_failure()])?
                .commit_from_file(model)?)
        })();
        match attempt {
            Ok(session) => return Ok((session, format!("GPU (DirectML adapter {adapter})"))),
            Err(error) => eprintln!("DirectML unavailable, falling back to CPU: {error:#}"),
        }
    }
    // Keep the existing app's fp16 CPU optimization workaround.
    Ok((
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level2)?
            .commit_from_file(model)?,
        if gpu {
            "CPU (DirectML unavailable)".into()
        } else {
            "CPU".into()
        },
    ))
}
