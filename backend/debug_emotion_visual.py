import cv2
import numpy as np
from utils.face_detector import FaceDetector
from utils.emotion_predictor import EmotionPredictor

def debug_emotion():
    """Visual debug to see what the emotion model receives"""
    
    detector = FaceDetector()
    predictor = EmotionPredictor()
    
    cap = cv2.VideoCapture(0)
    
    print("="*60)
    print("🔍 EMOTION MODEL VISUAL DEBUGGER")
    print("="*60)
    print("This will show you EXACTLY what the emotion model sees")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if len(faces) > 0:
            # Get first face
            (x, y, w, h) = faces[0]
            
            # Extract face
            face_img = frame[y:y+h, x:x+w]
            
            # Show what emotion model will receive
            # 1. Convert to grayscale (like the model does)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # 2. Resize to 64x64 (model input size)
            resized_face = cv2.resize(gray_face, (64, 64))
            
            # 3. Get emotion prediction
            emotion, conf, all_scores = predictor.predict_emotion(face_img)
            
            # Display original face
            face_display = cv2.resize(face_img, (200, 200))
            
            # Display grayscale face
            gray_display = cv2.resize(gray_face, (200, 200))
            gray_display = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR)
            
            # Display 64x64 input
            input_display = cv2.resize(resized_face, (200, 200))
            input_display = cv2.cvtColor(input_display, cv2.COLOR_GRAY2BGR)
            
            # Create combined display
            row1 = np.hstack([face_display, gray_display, input_display])
            
            # Add labels
            cv2.putText(row1, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(row1, "Grayscale", (210, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(row1, "Model Input (64x64)", (410, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show predictions
            pred_img = np.zeros((200, 600, 3), dtype=np.uint8)
            y_offset = 30
            
            cv2.putText(pred_img, f"Prediction: {emotion} ({conf:.3f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
            
            # Show all emotions sorted
            sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            for emotion_name, score in sorted_emotions:
                color = (0, 255, 0) if score > 0.3 else (255, 255, 255)
                cv2.putText(pred_img, f"{emotion_name}: {score:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
            
            # Combine all
            combined = np.vstack([row1, pred_img])
            
            cv2.imshow('Emotion Debug (Q to quit)', combined)
        else:
            cv2.imshow('Emotion Debug (Q to quit)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_emotion()
