import sys
import platform

# Platform-specific setup dispatcher
system = platform.system().lower()

if system == 'windows':
    print("Detected Windows platform. Using setup_windows.py")
    exec(open('setup_windows.py').read())
elif system == 'darwin':
    print("Detected macOS platform. Using setup_macos.py")
    exec(open('setup_macos.py').read())
elif system == 'linux':
    print("Detected Linux platform. Using setup_linux.py")
    exec(open('setup_linux.py').read())
else:
    raise RuntimeError(f"Unsupported platform: {system}. Please use setup_windows.py, setup_macos.py, or setup_linux.py manually.")
