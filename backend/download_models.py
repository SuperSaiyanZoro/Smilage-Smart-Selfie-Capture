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

# 2. Age Prediction Model (Caffe)
print("2. Age Prediction Model (Caffe)")
download_file(
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "models/age_net.caffemodel"
)

download_file(
    "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/deploy_age.prototxt",
    "models/age_deploy.prototxt"
)

# 3. Gender Prediction Model (Caffe)
print("3. Gender Prediction Model (Caffe)")
download_file(
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
    "models/gender_net.caffemodel"
)

download_file(
    "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/deploy_gender.prototxt",
    "models/gender_deploy.prototxt"
)

# 4. Emotion Detection Model (ONNX)
print("4. Emotion Detection Model (ONNX)")
download_file(
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "models/emotion-ferplus-8.onnx"
)

print("="*60)
print("✅ All models downloaded successfully!")
print("="*60)
print("\nModel files saved in 'models/' directory:")
print("  - haarcascade_frontalface_default.xml (Face Detection)")
print("  - age_net.caffemodel + age_deploy.prototxt (Age)")
print("  - gender_net.caffemodel + gender_deploy.prototxt (Gender)")
print("  - emotion-ferplus-8.onnx (Emotion)")
