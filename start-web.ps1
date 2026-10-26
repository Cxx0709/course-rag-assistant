param(
    [int]$Port = 8000,
    [string]$Model = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Host "Please run: uv sync --extra dev"
    exit 1
}
$env:NO_PROXY = "127.0.0.1,localhost"
if ($Model) {
    $env:COURSE_RAG_OLLAMA_CHAT_MODEL = $Model
    $env:COURSE_RAG_OLLAMA_CHAT_THINK = "false"
    $env:COURSE_RAG_OLLAMA_CHAT_NUM_CTX = "16384"
    $env:COURSE_RAG_OLLAMA_CHAT_NUM_PREDICT = "512"
    $env:COURSE_RAG_OLLAMA_CHAT_TIMEOUT = "120"
}
Write-Host ""
Write-Host "Course Notes: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop."
Write-Host ""
& $projectPython -X utf8 -m course_rag_assistant.web --port $Port
exit $LASTEXITCODE
