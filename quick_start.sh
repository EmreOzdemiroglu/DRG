#!/bin/bash
# DRG Quick Start Script

echo "🚀 DRG Proje Kurulum ve Çalıştırma"
echo "===================================="
echo ""

# Python versiyon kontrolü
echo "1️⃣  Python versiyon kontrolü..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"
echo ""

# Dependencies kontrolü
echo "2️⃣  Dependencies kontrolü..."
if python3 -c "import dspy" 2>/dev/null; then
    echo "   ✅ dspy kurulu"
else
    echo "   ❌ dspy bulunamadı - kuruluyor..."
    pip install dspy>=2.4.0
fi

if python3 -c "import litellm" 2>/dev/null; then
    echo "   ✅ litellm kurulu"
else
    echo "   ❌ litellm bulunamadı - kuruluyor..."
    pip install litellm>=1.0.0
fi
echo ""

# Proje kurulumu
echo "3️⃣  Proje kurulumu..."
pip install -e . > /dev/null 2>&1
echo "   ✅ Proje kuruldu"
echo ""

# API Key kontrolü
echo "4️⃣  API Key kontrolü..."
if [ -n "$GEMINI_API_KEY" ]; then
    echo "   ✅ GEMINI_API_KEY ayarlı"
    export DRG_MODEL=${DRG_MODEL:-"gemini/gemini-2.0-flash-exp"}
elif [ -n "$OPENAI_API_KEY" ]; then
    echo "   ✅ OPENAI_API_KEY ayarlı"
    export DRG_MODEL=${DRG_MODEL:-"openai/gpt-4o-mini"}
else
    echo "   ⚠️  API key bulunamadı"
    echo "   Mock mode ile devam edilecek"
fi
echo ""

# Test çalıştırma
echo "5️⃣  Pipeline test çalıştırılıyor..."
echo "   python3 examples/full_pipeline_example.py 1example"
echo ""
python3 examples/full_pipeline_example.py 1example

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "📚 Daha fazla bilgi için:"
echo "   - SETUP.md dosyasına bakın"
echo "   - examples/ klasöründeki örneklere bakın"
echo ""

