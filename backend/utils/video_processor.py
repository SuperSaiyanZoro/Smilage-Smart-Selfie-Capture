import cv2
import numpy as np
import base64
from datetime import datetime
import os
from .face_detector import FaceDetector
from .age_predictor import AgePredictor
from .gender_predictor import GenderPredictor
from .emotion_predictor import EmotionPredictor

class VideoProcessor:
    """
    Main video processing service that coordinates all AI models
    """
    
    def __init__(self):
        """Initialize all AI models"""
        print("🤖 Initializing Video Processor...")
        
        self.detector = FaceDetector()
        self.age_predictor = AgePredictor()
        self.gender_predictor = GenderPredictor()
        self.emotion_predictor = EmotionPredictor()
        
        # Settings
        self.smile_threshold = 0.5
        self.capture_dir = "captured_images"
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Frame counter for optimization
        self.frame_count = 0
        
        # Cached predictions
        self.last_predictions = {}
        
        print("✅ Video Processor initialized successfully!")
    
    def process_frame(self, frame):
        """
        Process a single frame and return predictions
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Predictions including age, gender, emotion, faces, etc.
        """
        self.frame_count += 1
        
        # Detect faces (fast, do every frame)
        faces = self.detector.detect_faces(frame)
        
        predictions = {
            "faces": [],
            "frame_number": self.frame_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process each face
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            # Skip too small faces
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue
            
            # Check image quality
            is_clear, blur_score = self.detector.check_blur(face_img)
            
            # Run predictions (optimize by running less frequently)
            try:
                # Run emotion every frame (needed for smile detection)
                emotion, emotion_conf, all_emotions = self.emotion_predictor.predict_emotion(face_img)
                
                # Run age/gender every 5 frames
                if self.frame_count % 5 == 0:
                    age_range, age_conf = self.age_predictor.predict_age(face_img)
                    gender, gender_conf = self.gender_predictor.predict_gender(face_img)
                    age_mid = self.age_predictor.get_age_midpoint(age_range)
                    
                    # Cache these predictions
                    self.last_predictions['age'] = age_range
                    self.last_predictions['age_mid'] = age_mid
                    self.last_predictions['age_conf'] = age_conf
                    self.last_predictions['gender'] = gender
                    self.last_predictions['gender_conf'] = gender_conf
                
                # Use cached predictions if available
                age_range = self.last_predictions.get('age', 'Unknown')
                age_mid = self.last_predictions.get('age_mid', 0)
                age_conf = self.last_predictions.get('age_conf', 0.0)
                gender = self.last_predictions.get('gender', 'Unknown')
                gender_conf = self.last_predictions.get('gender_conf', 0.0)
                
                # Check for smile
                happiness_score = all_emotions.get('happiness', 0.0)
                is_smiling = happiness_score > self.smile_threshold
                
                face_data = {
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "age": age_range,
                    "age_midpoint": int(age_mid),
                    "age_confidence": float(age_conf),
                    "gender": gender,
                    "gender_confidence": float(gender_conf),
                    "emotion": emotion,
                    "emotion_confidence": float(emotion_conf),
                    "happiness_score": float(happiness_score),
                    "is_smiling": is_smiling,
                    "is_clear": is_clear,
                    "blur_score": float(blur_score),
                    "all_emotions": {k: float(v) for k, v in all_emotions.items()}
                }
                
                predictions["faces"].append(face_data)
                
            except Exception as e:
                print(f"⚠️ Error processing face: {e}")
                continue
        
        return predictions
    
    def draw_predictions(self, frame, predictions):
        """
        Draw predictions on frame for visualization
        
        Args:
            frame: Input frame
            predictions: Predictions dict from process_frame
            
        Returns:
            Annotated frame
        """
        for face in predictions["faces"]:
            bbox = face["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            
            # Choose color based on smile
            color = (0, 255, 0) if face["is_smiling"] else (255, 0, 0)
            thickness = 3 if face["is_smiling"] else 2
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Prepare text
            texts = [
                f"Age: {face['age_midpoint']} {face['age']}",
                f"Gender: {face['gender']}",
                f"Emotion: {face['emotion']}",
                f"Happy: {face['happiness_score']:.2f}"
            ]
            
            if face["is_smiling"]:
                texts.append("😊 SMILING!")
            
            # Draw text
            y_offset = y - 10
            for i, text in enumerate(texts):
                y_pos = y_offset - (len(texts) - i) * 25
                cv2.putText(
                    frame, text, (x, max(y_pos, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )
        
        return frame
    
    def capture_selfie(self, frame, predictions):
        """
        Capture and save selfie
        
        Args:
            frame: Input frame
            predictions: Predictions dict
            
        Returns:
            dict: Info about captured image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selfie_{timestamp}.jpg"
        filepath = os.path.join(self.capture_dir, filename)
        
        # Draw predictions on frame
        annotated_frame = self.draw_predictions(frame.copy(), predictions)
        
        # Save image
        cv2.imwrite(filepath, annotated_frame)
        
        return {
            "filename": filename,
            "filepath": filepath,
            "timestamp": timestamp,
            "predictions": predictions
        }
    
    def encode_frame_to_base64(self, frame):
        """
        Encode frame to base64 for sending over WebSocket
        
        Args:
            frame: Input frame
            
        Returns:
            Base64 encoded string
        """
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    
    def set_smile_threshold(self, threshold):
        """Update smile detection threshold"""
        self.smile_threshold = max(0.0, min(1.0, threshold))
        return self.smile_threshold
