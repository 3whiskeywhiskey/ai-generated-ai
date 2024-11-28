#!/bin/bash

# Create project structure
mkdir -p src/llm_project/model \
         tests \
         configs \
         scripts \
         checkpoints \
         logs \
         llm_project

# Create __init__.py files
touch src/llm_project/__init__.py
touch src/llm_project/model/__init__.py

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .