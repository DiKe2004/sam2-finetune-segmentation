# Contributing to Bookshelf SAM2

Thanks for your interest in contributing! 🎉  
This project welcomes issues, pull requests, docs improvements, and ideas.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Propose a Change](#how-to-propose-a-change)
- [Development Setup](#development-setup)
- [Coding Style](#coding-style)
- [Testing & Linting](#testing--linting)
- [Pull Request Checklist](#pull-request-checklist)
- [Commit Messages](#commit-messages)
- [License](#license)

## Code of Conduct
By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Propose a Change
1. **Search existing issues** to avoid duplicates.
2. **Open a new issue** describing the problem or proposal.
3. For substantial changes, **discuss in the issue** before you start coding.
4. When ready, **open a pull request (PR)** referencing the issue.

## Development Setup
```bash
# clone
git clone https://github.com/<you>/<repo>.git
cd <repo>

# create env
conda env create -f environment.yml
conda activate sam2ocr

# install SAM2 (editable)
cd sam2 && pip install -e . && cd ..

# optional: dev tools
pip install -r requirements-dev.txt
