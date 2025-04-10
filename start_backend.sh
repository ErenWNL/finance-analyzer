#!/bin/bash

# Set environment variables
export FLASK_APP=backend/app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
export MODELS_DIR=$(pwd)/backend/models

# Create models directory if it doesn't exist
mkdir -p $MODELS_DIR

# Start Flask server
cd $(dirname "$0")
echo "Starting Flask server with models directory at $MODELS_DIR"
python -m flask run --host=0.0.0.0 --port=5001 