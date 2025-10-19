# Smilage - Smart Selfie Capture with AI-Based Analysis

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18.0+-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)

**AI-Based Image Analysis Tool for Smile Detection and Age Prediction**

A real-time web application that captures selfies with intelligent smile detection, age prediction, gender classification, and emotion recognition using computer vision and deep learning.

---

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [AI Models](#ai-models)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## ✨ Features

### Core Features
- 🎥 **Real-time Video Streaming** - WebSocket-based live camera feed
- 😊 **Smile Detection** - Haar Cascade-based smile recognition
- 👤 **Face Detection** - OpenCV Haar Cascade face detection
- 🎂 **Age Prediction** - Deep learning age estimation (8 age ranges)
- 👫 **Gender Classification** - Binary gender prediction
- 😄 **Emotion Recognition** - 8 emotion categories (neutral, happiness, surprise, etc.)
- 📸 **Image Capture** - Manual and automatic selfie capture
- 🖼️ **Gallery Management** - View, delete, and download captured images
- ⚙️ **Adjustable Settings** - Real-time smile threshold control
- 📊 **Performance Monitoring** - CPU and memory usage tracking

### Technical Features
- ⚡ Low-latency WebSocket communication
- 🔄 Real-time predictions (15-30 FPS)
- 💾 Persistent image storage
- 🎨 Modern, responsive UI
- 🔧 RESTful API endpoints
- 📱 Cross-platform compatibility

---

## 🎬 Demo

### Application Screenshots

**Main Interface**
- Real-time video feed with face detection
- Live predictions displayed on screen
- Green box indicates smiling detected
- Blue box indicates neutral expression

**Smile Detection in Action**
- Green box: Smiling detected ✅
- Blue box: Neutral expression

**Gallery View**
- Captured selfies with predictions
- Delete and download options

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Modern web framework
- **OpenCV** - Computer vision library
- **ONNX Runtime** - AI model inference
- **Uvicorn** - ASGI server
- **NumPy** - Numerical computing

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **JavaScript (ES6+)**
- **CSS3** - Styling

### AI/ML Models
- **Haar Cascades** - Face and smile detection
- **Caffe Models** - Age and gender prediction
- **FER+ ONNX** - Emotion recognition

---

## 🏗️ System Architecture

```
┌─────────────────┐          WebSocket           ┌──────────────────┐
│                 │◄────────────────────────────►│                  │
│  React Frontend │                               │  FastAPI Backend │
│                 │          REST API             │                  │
│  - Video Feed   │◄────────────────────────────►│  - Face Detect   │
│  - Gallery UI   │                               │  - Age Predict   │
│  - Settings     │                               │  - Gender Pred   │
└─────────────────┘                               │  - Smile Detect  │
                                                  │  - Emotion Rec   │
                                                  └──────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │   AI Models      │
                                                  │  - Haar Cascade  │
                                                  │  - Caffe Models  │
                                                  │  - ONNX Models   │
                                                  └──────────────────┘
```

---

## 📥 Installation

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** and npm
- **Git**
- **Webcam** (for testing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Smilage-Smart-Selfie-Capture.git
cd Smilage-Smart-Selfie-Capture
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download AI models
python download_models.py
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
```

---

## 🚀 Usage

### Starting the Application

**Terminal 1 - Backend Server:**
```bash
cd backend
venv\Scripts\activate  # Windows
python main.py
```
Backend runs on: `http://localhost:8000`

**Terminal 2 - Frontend Server:**
```bash
cd frontend
npm run dev
```
Frontend runs on: `http://localhost:3000`

### Using the Application

1. **Open Browser** - Navigate to `http://localhost:3000`
2. **Start Camera** - Click "▶ Start Camera" button
3. **Position Face** - Ensure your face is visible in the frame
4. **Smile Detection** - Smile to see green box and "SMILING!" indicator
5. **Capture Selfie** - Click "📸 Capture" button
6. **View Gallery** - See captured images in the sidebar
7. **Adjust Settings** - Use slider to modify smile detection threshold
8. **Stop Camera** - Click "⏹ Stop Camera" when done

### API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `GET /api/system-info` - System performance metrics
- `GET /api/gallery` - Get all captured images
- `DELETE /api/gallery/{filename}` - Delete specific image
- `DELETE /api/gallery` - Clear all images
- `POST /api/settings/smile-threshold` - Update smile threshold
- `WS /ws` - WebSocket for video streaming

### API Documentation

Interactive API docs available at: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
Smilage-Smart-Selfie-Capture/
│
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── requirements.txt           # Python dependencies
│   ├── download_models.py         # Model download script
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── face_detector.py       # Face detection
│   │   ├── age_predictor.py       # Age prediction
│   │   ├── gender_predictor.py    # Gender classification
│   │   ├── emotion_predictor.py   # Emotion recognition
│   │   ├── smile_detector.py      # Smile detection
│   │   └── video_processor.py     # Main processing pipeline
│   ├── models/                    # AI model files
│   └── captured_images/           # Saved selfies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   ├── App.css               # Component styles
│   │   ├── index.css             # Global styles
│   │   └── main.jsx              # Entry point
│   ├── package.json              # Node dependencies
│   └── vite.config.js            # Vite configuration
│
├── docs/                         # Documentation
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

---

## 🤖 AI Models

### Face Detection
- **Model**: Haar Cascade Frontal Face
- **Source**: OpenCV
- **Accuracy**: ~85-90%
- **Speed**: Real-time (30+ FPS)

### Smile Detection
- **Model**: Haar Cascade Smile
- **Source**: OpenCV
- **Method**: Pattern matching
- **Threshold**: 0.15 (configurable 0.00-0.50)
- **Performance**: 
  - Not Smiling: 0.00-0.10
  - Smiling: 0.15-0.60

### Age Prediction
- **Model**: Caffe Deep Learning Model
- **Source**: AgeGenderDeepLearning (Gil Levi)
- **Categories**: 8 age ranges
  - (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- **Accuracy**: ±5-7 years
- **Input**: 227x227 RGB face image

### Gender Classification
- **Model**: Caffe Deep Learning Model
- **Source**: AgeGenderDeepLearning (Gil Levi)
- **Categories**: Male, Female
- **Accuracy**: 85-90%
- **Input**: 227x227 RGB face image

### Emotion Recognition
- **Model**: FER+ ONNX Model
- **Source**: Microsoft Emotion Recognition
- **Categories**: 8 emotions
  - Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear, Contempt
- **Input**: 64x64 Grayscale face image
- **Note**: Used for emotion display only (not for smile detection)

---

## ⚠️ Known Limitations

### Model Limitations

1. **Age Prediction Accuracy**
   - Expected error: ±5-7 years
   - Performance varies with lighting and angle
   - Trained primarily on Western faces (IMDB/Wiki dataset)

2. **Gender Classification**
   - Binary classification only (Male/Female)
   - 85-90% accuracy on diverse faces
   - May have bias based on training data

3. **Emotion Recognition**
   - FER+ model shows high neutral scores
   - Best results with exaggerated expressions
   - Not used for smile detection (Haar Cascade used instead)

4. **Smile Detection**
   - Haar Cascade scores: typically 0.15-0.60 for smiles
   - Sensitive to lighting conditions
   - May require threshold adjustment per user

### Technical Limitations

- **Camera Access**: Requires webcam permission
- **Browser Compatibility**: Modern browsers only (Chrome, Firefox, Edge)
- **Performance**: Dependent on CPU power (no GPU acceleration)
- **Storage**: Local file system only (no cloud storage)
- **Single User**: One active camera session at a time

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Auto-capture on smile detection
- [ ] Multiple face detection and tracking
- [ ] Image filters and effects
- [ ] Export gallery as ZIP
- [ ] Confidence score visualization
- [ ] Dark/Light theme toggle
- [ ] Mobile responsive design improvements

### Technical Improvements
- [ ] GPU acceleration support
- [ ] Custom model training on diverse datasets
- [ ] Cloud storage integration
- [ ] User authentication and profiles
- [ ] Real-time statistics dashboard
- [ ] Video recording capability
- [ ] Batch processing mode

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Follow Airbnb JavaScript Style Guide
- **Comments**: Write clear, concise comments
- **Documentation**: Update README for new features

---

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **FastAPI** - Modern Python web framework
- **React** - UI framework
- **Gil Levi** - Age and Gender prediction models
- **Microsoft** - FER+ emotion recognition model
- **ONNX** - AI model format and runtime

---

## 📊 Project Statistics

- **Lines of Code**: ~2,500+
- **Development Time**: 2 weeks
- **Technologies Used**: 10+
- **AI Models**: 5
- **API Endpoints**: 8+

---
