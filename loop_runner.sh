#!/bin/bash

PROXY_URL=$1

if [ -z "$PROXY_URL" ]; then
  echo "Error: Please provide a proxy URL."
  exit 1
fi

while true; do
  # Check for stop file
  if [ -f "stop" ]; then
    echo "🛑 'stop' file detected. Stopping loop gracefully."
    rm "stop" # Remove the file so the script works next time
    exit 0
  fi

  echo "Running command..."
  runpod_runner \
    --proxy "$PROXY_URL" \
    --length 81

  echo "Command finished. Restarting in 1 second..."
  sleep 1
done
