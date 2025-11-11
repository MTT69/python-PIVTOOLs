#!/usr/bin/env python
"""
Test script to verify PIVTOOLs package installation and structure
"""
import sys
import importlib.util

def check_module(module_name, description):
    """Check if a module can be imported"""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {description}: Module '{module_name}' not found")
        return False
    else:
        print(f"✓ {description}: Module '{module_name}' found at {spec.origin}")
        return True

def check_entry_point(command_name):
    """Check if an entry point is available"""
    import shutil
    if shutil.which(command_name):
        print(f"✓ Command '{command_name}' is available")
        return True
    else:
        print(f"❌ Command '{command_name}' not found in PATH")
        return False

def main():
    print("=" * 60)
    print("PIVTOOLs Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check core modules
    print("Checking Core Modules:")
    all_ok &= check_module("pivtools_core", "Core package")
    all_ok &= check_module("pivtools_core.config", "Core config")
    all_ok &= check_module("pivtools_core.paths", "Core paths")
    print()
    
    # Check CLI modules
    print("Checking CLI Modules:")
    all_ok &= check_module("pivtools_cli", "CLI package")
    all_ok &= check_module("pivtools_cli.cli", "CLI main module")
    print()
    
    # Check GUI modules
    print("Checking GUI Modules:")
    all_ok &= check_module("pivtools_gui", "GUI package")
    all_ok &= check_module("pivtools_gui.app", "GUI app module")
    print()
    
    # Check entry points
    print("Checking Entry Points:")
    all_ok &= check_entry_point("pivtools-cli")
    all_ok &= check_entry_point("pivtools-gui")
    print()
    
    # Try importing the main functions
    print("Checking Main Functions:")
    try:
        from pivtools_cli.cli import main as cli_main
        print("✓ CLI main function can be imported")
    except ImportError as e:
        print(f"❌ Cannot import CLI main function: {e}")
        all_ok = False
    
    try:
        from pivtools_gui.app import main as gui_main
        print("✓ GUI main function can be imported")
    except ImportError as e:
        print(f"❌ Cannot import GUI main function: {e}")
        all_ok = False
    print()
    
    # Final verdict
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! PIVTOOLs is properly installed.")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
