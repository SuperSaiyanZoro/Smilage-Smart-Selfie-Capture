import urllib.request
import os

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded: {filename}\n")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}\n")

# Create models directory
os.makedirs("models", exist_ok=True)

print("="*60)
print("📦 Downloading Pre-trained Models for Smilage")
print("="*60 + "\n")

# 1. Haar Cascade for Face Detection
print("1. Haar Cascade Face Detector")
download_file(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "models/haarcascade_frontalface_default.xml"
)

# NEW: Haar Cascade for Smile Detection
print("2. Haar Cascade Smile Detector")
download_file(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml",
    "models/haarcascade_smile.xml"
)

# 3. Age Prediction Model (Caffe)
print("3. Age Prediction Model (Caffe)")
download_file(
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "models/age_net.caffemodel"
)

# 4. Gender Prediction Model (Caffe)
print("4. Gender Prediction Model (Caffe)")
download_file(
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
    "models/gender_net.caffemodel"
)

# 5. Emotion Detection Model (ONNX) - Keep for emotion, not smile
print("5. Emotion Detection Model (ONNX)")
download_file(
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "models/emotion-ferplus-8.onnx"
)

print("="*60)
print("✅ All models downloaded successfully!")
print("="*60)
