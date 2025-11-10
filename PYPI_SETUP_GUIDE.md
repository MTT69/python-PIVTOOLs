# PyPI Publishing Guide for PIVTOOLs

This guide walks you through setting up automated PyPI publishing for the PIVTOOLs packages using GitHub Actions.

## 📋 Prerequisites

### 1. PyPI Account and API Token
1. Create a PyPI account at https://pypi.org/account/register/
2. Generate an API token:
   - Go to https://pypi.org/manage/account/token/
   - Create a new token with scope "Entire account"
   - Copy the token (you won't be able to see it again!)

### 2. GitHub Repository
Ensure your repository has:
- The package structure we created (`pivtools_core/`, `pivtools_cli/`, `pivtools_gui/`)
- The GitHub Actions workflow (`.github/workflows/publish.yml`)
- All necessary files committed and pushed

## 🔧 GitHub Repository Setup

### 1. Add PyPI API Token to GitHub Secrets
1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI API token
6. Click **Add secret**

### 2. Verify Repository Structure
Ensure your repository contains:
```
.github/
  workflows/
    publish.yml
pivtools_core/
  pyproject.toml
  pivtools_core/
    __init__.py
    config.py
    paths.py
    ...
pivtools_cli/
  pyproject.toml
  setup.py
  MANIFEST.in
  pivtools_cli/
    ...
pivtools_gui/
  pyproject.toml
  pivtools_gui/
    ...
README.md
```

## 🚀 Publishing Process

### 1. Create a Release on GitHub
1. Go to your GitHub repository
2. Click **Releases** in the right sidebar
3. Click **Create a new release**
4. Choose a tag version (e.g., `v0.1.0`)
5. Title: `Release v0.1.0`
6. Description: Describe what's new in this release
7. **Important**: Check **Set as a pre-release** if this is not a stable release
8. Click **Publish release**

### 2. Monitor the GitHub Actions Workflow
1. Go to the **Actions** tab in your repository
2. You should see a workflow run for "Publish to PyPI"
3. Click on the workflow run to monitor progress

The workflow will:
- Build wheels for Windows, Linux, and macOS
- Compile C extensions with FFTW
- Test basic imports
- Publish packages to PyPI in order: core → CLI → GUI

## 📦 Package Dependencies

The packages are published in dependency order:

1. **`pivtools-core`** (no dependencies)
2. **`pivtools-cli`** (depends on `pivtools-core`)
3. **`pivtools-gui`** (depends on `pivtools-core` and `pivtools-cli`)

## 🧪 Testing the Setup

### 1. Test Local Installation
Before publishing, test that your packages can be installed:

```bash
# Create a test environment
python -m venv test_env
test_env\Scripts\activate  # On Windows
# source test_env/bin/activate  # On Linux/macOS

# Install from local builds
pip install pivtools_core/dist/pivtools_core-0.1.0-py3-none-any.whl
pip install pivtools_cli/dist/pivtools_cli-0.1.0-py3-none-any.whl
pip install pivtools_gui/dist/pivtools_gui-0.1.0-py3-none-any.whl

# Test imports
python -c "import pivtools_core; print('Core OK')"
python -c "import pivtools_cli; print('CLI OK')"
python -c "import pivtools_gui; print('GUI OK')"
```

### 2. Test PyPI Installation
After publishing, test installation from PyPI:

```bash
# Create a fresh environment
python -m venv test_pypi
test_pypi\Scripts\activate  # On Windows

# Install from PyPI
pip install pivtools-core pivtools-cli pivtools-gui

# Test the commands
pivtools-cli --help
pivtools-gui  # Should start the web server
```

## 🔍 Monitoring and Troubleshooting

### Check Build Status
- Go to **Actions** tab → Click on the workflow run
- Check each job (build-wheels, build-sdist, publish)
- Look at the logs for any errors

### Common Issues

#### 1. Build Failures
**Problem**: C extension compilation fails
**Solution**:
- Check that FFTW environment variables are set correctly
- Ensure the C source files are included in `MANIFEST.in`
- Verify that `setup.py` can find the source files

#### 2. Import Errors
**Problem**: Packages can't import each other
**Solution**:
- Check that dependencies are correctly specified in `pyproject.toml`
- Ensure import statements use the correct package names
- Verify that `__init__.py` files exist

#### 3. PyPI Upload Failures
**Problem**: Authentication or permission errors
**Solution**:
- Verify the `PYPI_API_TOKEN` secret is set correctly
- Check that you have maintainer/owner permissions on PyPI
- Ensure package names don't conflict with existing packages

#### 4. Missing Files in Distribution
**Problem**: Important files not included in the package
**Solution**:
- Check `MANIFEST.in` for CLI package
- Verify `pyproject.toml` includes all necessary files
- Test with `python -m build` locally first

### Logs and Debugging
- **Build logs**: Check the "Build wheels" job output
- **Test logs**: Look at the import test commands
- **PyPI logs**: Check the "Publish to PyPI" job

## 📋 Workflow Configuration Details

### Environment Variables
The workflow sets up FFTW paths for C compilation:
- **Linux**: `/usr/include` and `/usr/lib/x86_64-linux-gnu`
- **macOS**: `/opt/homebrew/include` and `/opt/homebrew/lib`
- **Windows**: `C:\fftw\include` and `C:\fftw\lib`

### Supported Platforms
- **Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Architectures**: 64-bit only (auto64)
- **Operating Systems**: Ubuntu, Windows, macOS

### Build Tools
- **cibuildwheel**: For cross-platform wheel building
- **setuptools**: For package configuration
- **build**: For source distribution creation

## 🎯 Best Practices

### Version Management
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version numbers in all `pyproject.toml` files
- Keep versions synchronized across packages

### Release Process
1. Update version numbers
2. Update CHANGELOG.md
3. Commit changes
4. Create GitHub release
5. Monitor Actions workflow
6. Verify PyPI publication

### Security
- Never commit API tokens to code
- Use GitHub secrets for sensitive data
- Regularly rotate API tokens
- Limit token scope when possible

## 📞 Support

If you encounter issues:
1. Check the GitHub Actions logs
2. Verify your PyPI account permissions
3. Test locally before publishing
4. Check existing GitHub issues for similar problems

## 📝 Additional Resources

- [PyPI Documentation](https://pypi.org/help/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [cibuildwheel Documentation](https://cibuildwheel.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Happy publishing! 🚀**</content>
<parameter name="filePath">c:\Users\mtt1e23\OneDrive - University of Southampton\Documents\python-PIVTOOLs\PYPI_SETUP_GUIDE.md