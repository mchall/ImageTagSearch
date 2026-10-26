# CLIP model assets

The search app uses the Xenova ONNX export of OpenAI CLIP ViT-B/32:
https://huggingface.co/Xenova/clip-vit-base-patch32

Required files: `text_model.onnx`, `vision_model.onnx`, `vocab.json`, `merges.txt`.
The two ONNX files total about 606 MB and are excluded from Git. Run
`./setup-clip.ps1`, or copy the files from your existing ml_playground:

```powershell
./setup-clip.ps1 -SourceDirectory C:/Users/user-pc/Desktop/ml_playground/Downloads/CLIP
```

The model repository declares the MIT license. Preserve upstream attribution
when redistributing. Tokenizer data is distributed with the same model repository.
The paired encoders must produce matching projected embeddings (`text_embeds`
and `image_embeds`). DINOv2 models are not compatible with text search.
