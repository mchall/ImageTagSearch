param([string]$SourceDirectory)
$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot 'Model/CLIP'
New-Item -ItemType Directory -Force $destination | Out-Null
foreach ($name in @('text_model.onnx', 'vision_model.onnx', 'vocab.json', 'merges.txt')) {
    $target = Join-Path $destination $name
    if ($SourceDirectory) {
        Copy-Item -LiteralPath (Join-Path $SourceDirectory $name) -Destination $target
    } elseif (!(Test-Path -LiteralPath $target)) {
        $remote = if ($name.EndsWith('.onnx')) { "onnx/$name" } else { $name }
        Invoke-WebRequest -Uri "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/$remote" -OutFile $target
    }
}
Write-Host "CLIP assets ready: $destination"
