# SigLIP2 model assets

Upstream: [onnx-community/siglip2-base-patch16-224-ONNX](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX), converted from Google's SigLIP2 base model.

Required files: `text_model_fp16.onnx`, `vision_model_fp16.onnx`, `tokenizer.json`.
Run `./setup-clip.ps1` from the repository root, or supply `-SourceDirectory` with a folder containing those files. Select another asset folder under **Model settings** if needed. ONNX weights are excluded from Git.

The app uses `input_ids` (int64, 1×64), `pixel_values` (float32, 1×3×224×224), and the paired `pooler_output` embeddings (the text pooler includes its projection). Internal fp16 weights do not imply fp16 input tensors. The tokenizer vocabulary and image preprocessing must match the encoders; old CLIP assets are incompatible.

The upstream repository declares Apache-2.0. Preserve upstream notices and license terms when redistributing model assets.
