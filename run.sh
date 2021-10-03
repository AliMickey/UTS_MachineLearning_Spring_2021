#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Setting up environment"
    echo "python3-venv package is required"
    sudo apt-get install python3-venv
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "Setup complete, running main.py"
    python3 main.py
else
    echo "Running main.py"
    source venv/bin/activate
    python3 main.py
fi