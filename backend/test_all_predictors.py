import cv2
from utils.face_detector import FaceDetector
from utils.age_predictor import AgePredictor
from utils.gender_predictor import GenderPredictor
from utils.emotion_predictor import EmotionPredictor

def test_all_predictors():
    """Test face detection with age, gender, and emotion prediction"""
    
    print("="*60)
    print("🤖 Testing All AI Predictors with Webcam")
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
    print("\nInstructions:")
    print("  - Position your face in front of the camera")
    print("  - See real-time predictions for age, gender, and emotion")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save a snapshot\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        return
    
    print("📸 Starting predictions... (Press 'q' to quit, 's' to save)\n")
    
    snapshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Make predictions
            try:
                age_range, age_conf = age_predictor.predict_age(face_img)
                gender, gender_conf = gender_predictor.predict_gender(face_img)
                emotion, emotion_conf, _ = emotion_predictor.predict_emotion(face_img)
                
                # Get midpoint age
                age_mid = age_predictor.get_age_midpoint(age_range)
                
                # Check if smiling
                is_smiling = emotion_predictor.is_smiling(emotion, emotion_conf)
                smile_text = "😊 SMILING!" if is_smiling else ""
                
                # Draw rectangle around face
                color = (0, 255, 0) if is_smiling else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Prepare text
                text_lines = [
                    f"Age: {age_mid} {age_range}",
                    f"Gender: {gender} ({gender_conf:.2f})",
                    f"Emotion: {emotion} ({emotion_conf:.2f})",
                    smile_text
                ]
                
                # Draw text above face
                y_offset = y - 10
                for i, text in enumerate(text_lines):
                    if text:  # Skip empty lines
                        y_pos = y_offset - (len(text_lines) - i) * 25
                        cv2.putText(
                            frame,
                            text,
                            (x, max(y_pos, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
        
        # Add instructions on frame
        cv2.putText(
            frame,
            "Press 'Q' to quit | 'S' to save snapshot",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Show frame
        cv2.imshow('Smilage - AI Predictions (Q=quit, S=save)', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_count += 1
            filename = f"snapshot_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    test_all_predictors()
