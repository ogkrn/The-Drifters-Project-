#!/bin/bash

# Render.com startup script for Streamlit app
# This ensures the app starts correctly on Render's infrastructure

echo "🚀 Starting Exoplanet Detection System..."
echo "📦 Python version: $(python --version)"
echo "📍 Working directory: $(pwd)"

# Create necessary directories
mkdir -p models
mkdir -p data

# List files to verify structure
echo "📂 Directory structure:"
ls -la

# Check if model exists
if [ -f "models/exoplanet_ml_model_hybrid.pkl" ]; then
    echo "✅ ML model found"
else
    echo "⚠️  ML model not found - app may fail"
fi

# Start Streamlit with proper configuration
streamlit run src/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --browser.gatherUsageStats=false
