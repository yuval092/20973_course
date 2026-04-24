#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt install -y stockfish

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Done! Run 'python main.py' to start."
