# CLIP model assets

The search app uses the [Xenova ONNX export of OpenAI CLIP ViT-B/32](https://huggingface.co/Xenova/clip-vit-base-patch32).

Required files: `text_model.onnx`, `vision_model.onnx`, `vocab.json`, `merges.txt`.
The two ONNX files total about 606 MB and are excluded from Git. From the
repository root, run:

```powershell
./setup-clip.ps1
```

The script downloads the encoders from the model repository's `onnx/` directory
and the tokenizer files from its root into this project's `Model/CLIP/` folder.
An internet connection is required for downloading; existing files are skipped.
No other project is required, and searches run locally after setup.

To reuse a local copy, supply a folder containing all four required files:

```powershell
./setup-clip.ps1 -SourceDirectory "C:/path/to/CLIP"
```

Launch the app from the repository root with `cargo run --locked`. If you keep
the assets elsewhere, select that folder under **Model settings** in the app.

The model repository declares the MIT license. Preserve upstream attribution
when redistributing. Tokenizer data is distributed with the same model repository.
The paired encoders must produce matching projected embeddings (`text_embeds`
and `image_embeds`). DINOv2 models are not compatible with text search.
