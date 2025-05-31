#!/usr/bin/env bash


DET_IOU_THRESHOLD=dynamic
PORT=8192
WORKERS=16 # Note: Uvicorn WORKERS usually means processes within *one* instance.

JOB_ID="j-$(cat /dev/urandom | tr -dc 'a-z0-9' | head -c 10)"
echo "Generated Short Job ID: ${JOB_ID}"

STORAGE_PATH="$(pwd)/.reward_server" # Use absolute path
JOB_DIR="$STORAGE_PATH/$JOB_ID" # Directory for this specific job

echo "Checking and creating job directory (if needed): $JOB_DIR"
mkdir -p "$JOB_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Could not create job directory $JOB_DIR. Please check permissions or path."
  exit 1
fi

# Change to the directory where the FastAPI app is located
cd ./reward_server

uvicorn reward_serving_fastapi:app \
  --host 0.0.0.0  \
  --port $PORT \
  --workers $WORKERS & # Run Uvicorn in the background

# Get the PID of the Uvicorn background process
UVICORN_PID=$!
echo "Uvicorn server started in the background with PID: $UVICORN_PID"

# Wait a few seconds to ensure the server has enough time to start and bind to the port
echo "Waiting 5 seconds for the server to fully start..."
sleep 5

# --- Get IP Address ---
SERVER_IP=""

# Attempt 1: Use hostname -I to get the first non-loopback IP (usually works on Linux)
echo "Attempting to get IP address using 'hostname -I'..."
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

# Attempt 2: If hostname -I fails or doesn't return an IP, try 'ip addr' (Linux)
if [ -z "$SERVER_IP" ]; then
  echo "'hostname -I' failed to get IP address, trying 'ip addr'..."
  SERVER_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -Ev '^(127\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' | head -n 1 2>/dev/null) # Try to get a public IP
  if [ -z "$SERVER_IP" ]; then # If no public IP, try to get the first private IP (not 127.0.0.1)
     SERVER_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v "127.0.0.1" | head -n 1 2>/dev/null)
  fi
fi

# Final check and fallback
if [ -z "$SERVER_IP" ]; then
  echo "Warning: Could not automatically determine a publicly accessible IP address. Using 0.0.0.0"
  SERVER_IP="0.0.0.0" # Or you could set this to 127.0.0.1 if primarily for local access
fi

echo "Determined server IP: $SERVER_IP"
echo "Server port: $PORT"

# --- Create empty file with IP:PORT as its name ---
FILENAME="$SERVER_IP:$PORT"
OUTPUT_PATH="$JOB_DIR/$FILENAME"

echo "Creating empty file at: $OUTPUT_PATH"
touch "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
  echo "Successfully created file $OUTPUT_PATH"
else
  echo "Error: Could not create file $OUTPUT_PATH."
fi

echo "---------------------------------------------------------------------"
echo "Uvicorn server (PID: $UVICORN_PID) is running in the background."
echo "To stop the server, you can use 'kill $UVICORN_PID' or press Ctrl+C in this terminal."
echo "---------------------------------------------------------------------"

# Wait for the Uvicorn process to end
wait $UVICORN_PID

echo "Uvicorn server (PID: $UVICORN_PID) has stopped."
