#!/bin/bash
# DRG API Server Başlatma Script'i
# Kullanım:
#   ./start_api_server.sh              # 1example (varsayılan)
#   ./start_api_server.sh 3            # 3example
#   DRG_EXAMPLE=4example ./start_api_server.sh  # 4example (environment variable ile)

# Gemini API Key (gerekli)
export GEMINI_API_KEY="${GEMINI_API_KEY:-AIzaSyBihxuHjN1hN0D0CzXdiYTEQxzjqtIoL9k}"

# Example seçimi: command line arg > environment variable > default
EXAMPLE_ARG="${1:-}"
if [ -n "$EXAMPLE_ARG" ]; then
    export DRG_EXAMPLE="${EXAMPLE_ARG}"
elif [ -z "$DRG_EXAMPLE" ]; then
    export DRG_EXAMPLE="1example"
fi

# Sayı formatını düzelt (3 -> 3example)
if [[ "$DRG_EXAMPLE" =~ ^[0-9]+$ ]]; then
    export DRG_EXAMPLE="${DRG_EXAMPLE}example"
fi

echo "🚀 DRG API Server başlatılıyor..."
echo "📌 Example: $DRG_EXAMPLE"
echo "🌐 URL: http://localhost:8000"
echo ""

cd "$(dirname "$0")"
python3 examples/api_server_example.py "$DRG_EXAMPLE"
