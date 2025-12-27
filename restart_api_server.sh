#!/bin/bash

# DRG API Server Restart Script
# Kullanım: ./restart_api_server.sh [example_name]

EXAMPLE_NAME=${1:-4example}

echo "🛑 Eski API server'ları durduruluyor..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 2

echo "🚀 Yeni API server başlatılıyor: $EXAMPLE_NAME"
export GEMINI_API_KEY="AIzaSyBihxuHjN1hN0D0CzXdiYTEQxzjqtIoL9k"
python3 examples/api_server_example.py "$EXAMPLE_NAME"

