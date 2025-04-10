#!/bin/bash

# Kill any existing Flask process
echo "Stopping existing Flask server..."
pkill -f "python.*app.py" || echo "No Flask server running"

# Start the Flask server in the background with output to a log file
echo "Starting Flask server..."
cd "$(dirname "$0")/backend"
python3 app.py > ../backend_output.log 2>&1 &

# Wait for the server to start
echo "Waiting for server to start..."
sleep 3

# Check if the server is running
if pgrep -f "python.*app.py" > /dev/null; then
    echo "Server started successfully. Check backend_output.log for details."
    echo "API will be available at http://localhost:5001"
else
    echo "Failed to start server. Check backend_output.log for errors."
fi 