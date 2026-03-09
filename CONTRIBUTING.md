# Contributing to whichllm

Thanks for your interest in contributing! Here's how you can help.

## Development Setup

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## How to Contribute

### Bug Reports
Open an issue with:
- Your hardware (GPU model, VRAM, OS)
- Python version
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
Open an issue describing the feature and why it would be useful.

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Submit a PR

### Adding GPU Support
To add a new GPU to the bandwidth database, edit `src/whichllm/constants.py` and add the GPU specs.

## Code Style
- Follow existing code conventions
- Use type hints
- Add tests for new functionality

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
