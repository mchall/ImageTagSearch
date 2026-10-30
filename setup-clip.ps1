param([string]$SourceDirectory)
$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot 'Model/SigLIP2'
New-Item -ItemType Directory -Force $destination | Out-Null
foreach ($name in @('text_model_fp16.onnx', 'vision_model_fp16.onnx', 'tokenizer.json')) {
    $target = Join-Path $destination $name
    if ($SourceDirectory) {
        Copy-Item -LiteralPath (Join-Path $SourceDirectory $name) -Destination $target
    } elseif (!(Test-Path -LiteralPath $target) -or (Get-Item -LiteralPath $target).Length -eq 0) {
        $remote = if ($name.EndsWith('.onnx')) { "onnx/$name" } else { $name }
        # Publish only completed downloads so an interrupted setup can be retried.
        $partial = "$target.partial"
        Invoke-WebRequest -Uri "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/$remote" -OutFile $partial
        Move-Item -LiteralPath $partial -Destination $target -Force
    }
}
Write-Host "SigLIP2 assets ready: $destination"
