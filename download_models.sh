#!/bin/bash
set -e

echo "================================"
echo "📦 Downloading Models from Google Drive"
echo "================================"

mkdir -p /app/models

download_from_gdrive() {
    FILE_ID=$1
    FILENAME=$2
    
    echo "⬇️  Downloading $FILENAME..."
    
    python3 << EOF
import gdown
import os
file_id = "$FILE_ID"
output_path = "/app/models/$FILENAME"
url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
gdown.download(url, output_path, quiet=False)
print(f"✅ {FILENAME} downloaded successfully")
EOF
}

if [ ! -f /app/models/densenet121.pth ]; then
    download_from_gdrive "1DF8_4hGTRLmNYHQVDTMP-KlQlPBlCA3x" "densenet121.pth"
fi

if [ ! -f /app/models/resnet152.pth ]; then
    download_from_gdrive "1WQ65QcYUar0U3WnFY93_gHbmhoG-5v5B" "resnet152.pth"
fi

echo "🎉 All models ready!"
ls -lh /app/models/
