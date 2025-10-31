import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from image_handling.load_images import read_pair

def test_set_reading():
    config = Config()
    print(f"Config loaded: time_resolved={config.time_resolved}")
    print(f"image_format: {config.image_format}")
    print(f"source_paths: {config.source_paths}")

    # Check if the .set file exists
    set_file_path = config.source_paths[0] / config.image_format
    print(f"Set file path: {set_file_path}")

    # Test reading one pair
    try:
        pair = read_pair(idx=1, camera_path=config.source_paths[0], camera=1, config=config)
        print(f"Successfully read pair 1: shape {pair.shape}, dtype {pair.dtype}")
        print(f"Frame A shape: {pair[0].shape}, Frame B shape: {pair[1].shape}")
    except Exception as e:
        print(f"Error reading pair: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_set_reading()