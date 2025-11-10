from pathlib import Path
from typing import Callable, Dict, List

from .generic_readers import read_png_jpeg, read_raw, read_tiff

# Registry for image readers
_READERS: Dict[str, Callable] = {}


def register_reader(extensions: List[str], reader_func: Callable):
    """Register an image reader for specific file extensions.

    Args:
        extensions: List of file extensions (e.g., ['.tiff', '.tif'])
        reader_func: Function that takes file_path and returns np.ndarray
    """
    for ext in extensions:
        _READERS[ext.lower()] = reader_func


def get_reader(file_path: str) -> Callable:
    """Get appropriate reader for a file based on its extension.

    Args:
        file_path: Path to the image file

    Returns:
        Reader function for the file type

    Raises:
        ValueError: If no reader is registered for the file type
    """
    ext = Path(file_path).suffix.lower()
    if ext not in _READERS:
        raise ValueError(f"No reader registered for file type: {ext}")
    return _READERS[ext]


def list_supported_formats() -> List[str]:
    """List all supported file formats."""
    return list(_READERS.keys())


# Import LaVision readers after functions are defined to avoid circular imports
from .lavision_reader import read_lavision_pair, read_lavision_ims_pair

# Register only lowercase variants for robustness
register_reader([".tiff", ".tif"], read_tiff)
register_reader([".png"], read_png_jpeg)
register_reader([".jpg", ".jpeg"], read_png_jpeg)
register_reader([".raw", ".cr2", ".nef", ".arw"], read_raw)
register_reader([".im7"], read_lavision_pair)
register_reader([".set"], read_lavision_ims_pair)