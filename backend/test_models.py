import os

print("="*60)
print("🔍 Checking Model Files")
print("="*60 + "\n")

models_dir = "models"
required_files = [
    "haarcascade_frontalface_default.xml",
    "age_net.caffemodel",
    "age_deploy.prototxt",
    "gender_net.caffemodel",
    "gender_deploy.prototxt",
    "emotion-ferplus-8.onnx"
]

all_present = True

for filename in required_files:
    filepath = os.path.join(models_dir, filename)
    exists = os.path.exists(filepath)
    
    if exists:
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✅ {filename:<40} ({size_mb:.2f} MB)")
    else:
        print(f"❌ {filename:<40} (MISSING)")
        all_present = False

print("\n" + "="*60)
if all_present:
    print("✅ All model files are present and ready!")
else:
    print("❌ Some model files are missing. Please check above.")
print("="*60)
