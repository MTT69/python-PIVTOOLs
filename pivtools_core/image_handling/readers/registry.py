from pathlib import Path
from typing import Callable, Dict, List

# Registry for image readers
_reader_registry: Dict[str, Callable] = {}


def register_reader(extensions: List[str], reader_func: Callable):
    """Register an image reader for specific file extensions."""
    for ext in extensions:
        _reader_registry[ext.lower()] = reader_func


def get_reader(file_path: str) -> Callable:
    """Get appropriate reader for a file based on its extension."""
    ext = Path(file_path).suffix.lower()
    if ext not in _reader_registry:
        raise ValueError(f"No reader registered for file type: {ext}")
    return _reader_registry[ext]


def list_supported_formats() -> List[str]:
    """List all supported file formats."""
    return list(_reader_registry.keys())
