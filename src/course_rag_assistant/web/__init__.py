"""Local Web app. Public imports and command entry points remain compatible."""
from .app import create_app, main
from .library import Document, Library

__all__ = ["Document", "Library", "create_app", "main"]
