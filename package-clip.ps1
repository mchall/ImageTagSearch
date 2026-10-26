param([string]$OutputDirectory = (Join-Path $PSScriptRoot 'dist'))
$ErrorActionPreference = 'Stop'
foreach ($name in @('text_model.onnx','vision_model.onnx','vocab.json','merges.txt')) {
    if (!(Test-Path -LiteralPath (Join-Path $PSScriptRoot "Model/CLIP/$name"))) { throw "Missing $name. Run ./setup-clip.ps1 first." }
}
cargo build --manifest-path (Join-Path $PSScriptRoot 'Cargo.toml') --release --locked --bin clip-search
if ($LASTEXITCODE -ne 0) { throw 'Rust release build failed' }
$destination = Join-Path $OutputDirectory 'ClipSearch'
New-Item -ItemType Directory -Force (Join-Path $destination 'Model/CLIP') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'target/release/clip-search.exe') -Destination $destination
foreach ($name in @('onnxruntime.dll','onnxruntime_providers_shared.dll','DirectML.dll')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "Runtime/win-x64/$name") -Destination $destination
}
foreach ($name in @('text_model.onnx','vision_model.onnx','vocab.json','merges.txt','README.md')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "Model/CLIP/$name") -Destination (Join-Path $destination 'Model/CLIP')
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Runtime/licenses') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'LICENSE'), (Join-Path $PSScriptRoot 'README.md') -Destination $destination
Compress-Archive -Path (Join-Path $destination '*') -DestinationPath (Join-Path $OutputDirectory 'ClipSearch-win-x64.zip') -Force
Write-Host "Release: $destination"
