import cv2
import time
from utils.face_detector import FaceDetector
from utils.age_predictor import AgePredictor
from utils.gender_predictor import GenderPredictor
from utils.emotion_predictor import EmotionPredictor

def test_fixed_predictions():
    """Fixed version with real-time emotion updates"""
    
    print("="*60)
    print("🔧 Testing FIXED AI Predictions")
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
    print("\n🔧 FIXES Applied:")
    print("  ✓ Emotion updates every 2 frames (real-time)")
    print("  ✓ Visual feedback shows when predictions update")
    print("  ✓ Debug info shows all emotion scores")
    print("  ✓ Better face preprocessing for age accuracy\n")
    print("📝 Note: Age models have ±5-7 years error - this is normal!")
    print("    They predict ranges, not exact ages.\n")
    print("Instructions:")
    print("  - Try different expressions")
    print("  - Watch happiness score change in real-time")
    print("  - Press 'd' to toggle debug mode")
    print("  - Press 'q' to quit\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        return
    
    print("📸 Starting predictions...\n")
    
    # Performance tracking
    frame_count = 0
    
    # Run emotion every 2 frames, age/gender every 10 frames
    emotion_interval = 2
    age_gender_interval = 10
    
    # Current predictions (initialize)
    current_age = "Analyzing..."
    current_gender = "Analyzing..."
    current_emotion = "neutral"
    current_emotion_conf = 0.0
    current_all_emotions = {}
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    fps_counter = 0
    
    # Debug mode
    debug_mode = False
    
    snapshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS every 30 frames
        if fps_counter >= 30:
            current_time = time.time()
            fps = fps_counter / (current_time - prev_time)
            prev_time = current_time
            fps_counter = 0
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Flags for what to update this frame
        update_emotion = (frame_count % emotion_interval == 0)
        update_age_gender = (frame_count % age_gender_interval == 0)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Check face size - skip if too small
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue
            
            # Update emotion (frequent)
            if update_emotion:
                try:
                    emotion, emotion_conf, all_emotions = emotion_predictor.predict_emotion(face_img)
                    
                    # Update current values
                    current_emotion = emotion
                    current_emotion_conf = emotion_conf
                    current_all_emotions = all_emotions
                    
                except Exception as e:
                    print(f"⚠️ Emotion prediction error: {e}")
            
            # Update age and gender (less frequent)
            if update_age_gender:
                try:
                    age_range, age_conf = age_predictor.predict_age(face_img)
                    gender, gender_conf = gender_predictor.predict_gender(face_img)
                    
                    age_mid = age_predictor.get_age_midpoint(age_range)
                    current_age = f"{age_mid} {age_range}"
                    current_gender = f"{gender} ({gender_conf:.2f})"
                    
                except Exception as e:
                    print(f"⚠️ Age/Gender prediction error: {e}")
            
            # Get happiness score
            happiness_score = current_all_emotions.get('happiness', 0.0)
            neutral_score = current_all_emotions.get('neutral', 0.0)
            
            # Improved smile detection
            is_smiling = False
            smile_method = ""
            
            # Method 1: High happiness score
            if happiness_score > 0.5:
                is_smiling = True
                smile_method = "High Happy"
            
            # Method 2: Happiness dominates neutral
            elif happiness_score > 0.3 and happiness_score > neutral_score:
                is_smiling = True
                smile_method = "Happy > Neutral"
            
            # Method 3: Check if emotion is happiness
            elif current_emotion == 'happiness' and current_emotion_conf > 0.4:
                is_smiling = True
                smile_method = "Emotion=Happy"
            
            # Draw rectangle
            color = (0, 255, 0) if is_smiling else (255, 0, 0)
            thickness = 4 if is_smiling else 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Prepare display text
            y_offset = y - 10
            line_height = 25
            
            # Basic info (always show)
            basic_info = [
                f"Age: {current_age}",
                f"Gender: {current_gender}",
                f"Emotion: {current_emotion} ({current_emotion_conf:.2f})",
                f"Happiness: {happiness_score:.2f} | Neutral: {neutral_score:.2f}"
            ]
            
            if is_smiling:
                basic_info.append(f"😊 SMILING! ({smile_method})")
            
            # Draw basic info
            for i, text in enumerate(basic_info):
                y_pos = y_offset - (len(basic_info) - i) * line_height
                font_scale = 0.7 if "SMILING" in text else 0.6
                color_text = (0, 255, 255) if "SMILING" in text else (0, 255, 0)
                thickness_text = 2
                
                cv2.putText(
                    frame, text, (x, max(y_pos, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color_text, thickness_text
                )
            
            # Debug mode - show all emotion scores
            if debug_mode:
                debug_y = y + h + 30
                cv2.putText(frame, "DEBUG - All Emotions:", (x, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                for idx, (emotion_name, score) in enumerate(sorted(current_all_emotions.items(), key=lambda x: x[1], reverse=True)):
                    debug_text = f"  {emotion_name}: {score:.3f}"
                    cv2.putText(frame, debug_text, (x, debug_y + 20 + idx*18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        # Status bar at bottom
        status_text = f"FPS: {fps:.1f} | Frame: {frame_count} | Press: Q=quit, S=save, D=debug"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update indicator (shows when predictions are updating)
        if update_emotion:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 255, 0), -1)
        
        # Show frame
        cv2.imshow('Smilage - Fixed AI (Q=quit, S=save, D=debug)', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_count += 1
            filename = f"snapshot_fixed_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 Saved: {filename}")
            print(f"   Emotion: {current_emotion} (conf: {current_emotion_conf:.2f})")
            print(f"   Happiness: {happiness_score:.3f}, Neutral: {neutral_score:.3f}")
            print(f"   All emotions: {current_all_emotions}\n")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"\n🔧 Debug mode: {'ON' if debug_mode else 'OFF'}\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print(f"📊 Final FPS: {fps:.1f}")
    print("="*60)

if __name__ == "__main__":
    test_fixed_predictions()
