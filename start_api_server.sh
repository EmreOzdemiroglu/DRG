#!/bin/bash
# DRG API Server Başlatma Script'i
# Kullanım:
#   ./start_api_server.sh              # 1example (varsayılan)
#   ./start_api_server.sh 3            # 3example
#   DRG_EXAMPLE=4example ./start_api_server.sh  # 4example (environment variable ile)

# Gemini API Key (gerekli)
# Güvenlik: burada key hardcode etmiyoruz. Environment variable veya .env üzerinden set edilmeli.
if [ -z "$GEMINI_API_KEY" ]; then
  echo "❌ GEMINI_API_KEY ayarlı değil. Lütfen environment variable veya .env ile ayarlayın."
  echo "   Örn: export GEMINI_API_KEY=\"your-gemini-api-key\""
  exit 1
fi

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
