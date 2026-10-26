use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64")
    {
        return;
    }
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let profile = out
        .ancestors()
        .nth(3)
        .expect("Cargo profile output directory");
    for name in [
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "DirectML.dll",
    ] {
        let source = root.join("Runtime/win-x64").join(name);
        println!("cargo:rerun-if-changed={}", source.display());
        let bytes = fs::read(&source).unwrap_or_else(|error| {
            panic!("Missing bundled runtime {}: {error}", source.display())
        });
        let destination = profile.join(name);
        if fs::read(&destination).ok().as_ref() != Some(&bytes) {
            fs::write(&destination, bytes).unwrap_or_else(|error| {
                panic!(
                    "Cannot copy {}: {error}. Close running development builds and retry.",
                    destination.display()
                )
            });
        }
    }
}
