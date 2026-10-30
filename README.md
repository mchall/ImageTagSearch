# Image Tag Search

Native Rust photo search using ONNX Runtime and egui/eframe. Pick a folder, enter a description, and browse ranked images. Searches run locally after setup.

![Screenshot](screenshot.jpg)

```powershell
./setup-clip.ps1
cargo run --locked
```

The app uses [SigLIP2 base patch16 224](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX), specifically `text_model_fp16.onnx` and `vision_model_fp16.onnx`, with the upstream `tokenizer.json`. Setup downloads these into `Model/SigLIP2/`. To copy an existing set, run `./setup-clip.ps1 -SourceDirectory "C:/path/to/SigLIP2"`.

Windows x64, stable Rust MSVC, Visual Studio C++ build tools and the Windows SDK are required. ONNX Runtime and DirectML DLLs are bundled under `Runtime/` and copied alongside the executable by `build.rs`.

- Recursive image search, ranked thumbnail grid, larger preview and navigation.
- First 50 matches shown initially; **Fetch next 50** reveals more.
- Background inference, progress, cancellation, and cached image and text embeddings.
- DirectML GPU adapter selection with CPU fallback and actual device reporting.
- JPEG, PNG, WebP, BMP and GIF (first frame); unreadable images are skipped.

Queries are lowercased, whitespace is canonicalized, and the upstream Gemma tokenizer adds EOS with truncation and right padding to 64 tokens. Images are converted to RGB, resized directly to 224×224 using bilinear filtering and normalized with mean/std 0.5. Rust and Pillow resampling can differ slightly. Normalized text and image embeddings are ranked by cosine similarity; scores are relative rankings, not confidence percentages.

Build a portable release with `./package-clip.ps1`; extract `dist/ClipSearch-win-x64.zip` and run `clip-search.exe`. Existing script and executable names are retained. Keep the model folder and runtime DLLs beside the executable. See [model details](Model/SigLIP2/README.md).

```powershell
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo run --locked --bin clip-search -- --smoke-test --folder "C:/path/to/photos"
```

The smoke test runs CPU inference for two queries, checks finite sorted cosine scores, and repeats search with cached image embeddings.
