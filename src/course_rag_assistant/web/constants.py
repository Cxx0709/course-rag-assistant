"""Web resource paths and upload policy."""
from pathlib import Path

STATIC = Path(__file__).parent / "static"
TEMPLATES = Path(__file__).parent / "templates"
MAX_UPLOAD = 50 * 1024 * 1024
SUPPORTED = {".pdf", ".pptx", ".md"}
