import cv2
import time
from utils.face_detector import FaceDetector
from utils.age_predictor import AgePredictor
from utils.gender_predictor import GenderPredictor
from utils.emotion_predictor import EmotionPredictor

def test_optimized_predictions():
    """Optimized test with better performance and smile detection"""
    
    print("="*60)
    print("🚀 Testing Optimized AI Predictions")
    print("="*60)
    print("\nInitializing models...\n")
    
    # Initialize all models
    try:
        detector = FaceDetector()
        age_predictor = AgePredictor()
        gender_predictor = GenderPredictor()
        emotion_predictor = EmotionPredictor()
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return
    
    print("\n✅ All models loaded successfully!")
    print("\nOptimizations enabled:")
    print("  ⚡ Predictions run every 5 frames (not every frame)")
    print("  ⚡ Lower smile detection threshold")
    print("  ⚡ FPS counter to monitor performance\n")
    print("Instructions:")
    print("  - Smile broadly to trigger detection")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save snapshot\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        return
    
    print("📸 Starting predictions...\n")
    
    # Performance tracking
    frame_count = 0
    prediction_interval = 5  # Run predictions every N frames
    fps_update_interval = 30
    
    # Cached predictions
    cached_age = "Unknown"
    cached_gender = "Unknown"
    cached_emotion = "neutral"
    cached_emotion_conf = 0.0
    cached_all_emotions = {}
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    
    snapshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % fps_update_interval == 0:
            current_time = time.time()
            fps = fps_update_interval / (current_time - prev_time)
            prev_time = current_time
        
        # Detect faces (this is fast, do every frame)
        faces = detector.detect_faces(frame)
        
        # Only run predictions every N frames (optimization)
        run_predictions = (frame_count % prediction_interval == 0)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Run predictions only periodically
            if run_predictions:
                try:
                    age_range, age_conf = age_predictor.predict_age(face_img)
                    gender, gender_conf = gender_predictor.predict_gender(face_img)
                    emotion, emotion_conf, all_emotions = emotion_predictor.predict_emotion(face_img)
                    
                    # Cache predictions
                    age_mid = age_predictor.get_age_midpoint(age_range)
                    cached_age = f"{age_mid} {age_range}"
                    cached_gender = f"{gender} ({gender_conf:.2f})"
                    cached_emotion = emotion
                    cached_emotion_conf = emotion_conf
                    cached_all_emotions = all_emotions
                    
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")
            
            # Check for smile using multiple methods
            is_smiling = False
            smile_reason = ""
            
            # Method 1: Direct happiness check with lower threshold
            if cached_emotion == 'happiness' and cached_emotion_conf > 0.3:
                is_smiling = True
                smile_reason = "Happy"
            
            # Method 2: Check happiness score even if not top emotion
            if cached_all_emotions.get('happiness', 0) > 0.25:
                is_smiling = True
                smile_reason = "Happy Score"
            
            # Method 3: Compare happiness vs neutral
            happiness_score = cached_all_emotions.get('happiness', 0)
            neutral_score = cached_all_emotions.get('neutral', 0)
            if happiness_score > 0.2 and happiness_score > neutral_score * 0.5:
                is_smiling = True
                smile_reason = "Happy > Neutral"
            
            # Draw rectangle (green if smiling, blue otherwise)
            color = (0, 255, 0) if is_smiling else (255, 0, 0)
            thickness = 3 if is_smiling else 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Prepare text
            smile_text = f"😊 SMILING! ({smile_reason})" if is_smiling else ""
            
            text_lines = [
                f"Age: {cached_age}",
                f"Gender: {cached_gender}",
                f"Emotion: {cached_emotion} ({cached_emotion_conf:.2f})",
                f"Happiness: {cached_all_emotions.get('happiness', 0):.2f}",
                smile_text
            ]
            
            # Draw text above face
            y_offset = y - 10
            for i, text in enumerate(text_lines):
                if text:
                    y_pos = y_offset - (len(text_lines) - i) * 25
                    # Use larger font and different color for smile text
                    font_scale = 0.7 if "SMILING" in text else 0.6
                    color_text = (0, 255, 255) if "SMILING" in text else (0, 255, 0)
                    cv2.putText(
                        frame,
                        text,
                        (x, max(y_pos, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color_text,
                        2
                    )
        
        # Show FPS and instructions
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Press 'Q' to quit | 'S' to save",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Show frame
        cv2.imshow('Smilage - Optimized AI (Q=quit, S=save)', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_count += 1
            filename = f"snapshot_optimized_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
            print(f"   Emotion: {cached_emotion}, Happiness: {cached_all_emotions.get('happiness', 0):.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print(f"📊 Average FPS: {fps:.1f}")
    print("="*60)

if __name__ == "__main__":
    test_optimized_predictions()
