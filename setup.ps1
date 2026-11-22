param([switch]$RulesOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache/uv'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'

function Confirm-Exit {
    if ($LASTEXITCODE -ne 0) { throw "Setup command failed with exit code $LASTEXITCODE." }
}

if (!(Test-Path '.venv/Scripts/python.exe')) {
    uv venv .venv --python 3.13
    Confirm-Exit
}
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
Confirm-Exit
& ./.venv/Scripts/python.exe -X utf8 -m finance_agent.filings
Confirm-Exit
& ./.venv/Scripts/python.exe -X utf8 -m finance_agent.corpus
Confirm-Exit
if (!$RulesOnly) {
    if (!(Test-Path '.model-venv/Scripts/python.exe')) {
        uv venv .model-venv --python 3.13
        Confirm-Exit
    }
    uv pip install --python .model-venv/Scripts/python.exe 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
    Confirm-Exit
    uv pip install --python .model-venv/Scripts/python.exe 'transformers==4.57.6' 'huggingface-hub[hf_xet]==0.36.2' fastapi uvicorn
    Confirm-Exit
    & ./.model-venv/Scripts/hf.exe download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --include '*.safetensors' '*.json' 'merges.txt' 'LICENSE'
    Confirm-Exit
}
