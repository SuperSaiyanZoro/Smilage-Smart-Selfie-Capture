import cv2
import numpy as np
import onnxruntime as ort

class EmotionPredictor:
    """
    Emotion prediction using pre-trained ONNX model
    """
    
    # Emotion labels from FER+ dataset
    EMOTION_LABELS = [
        'neutral', 'happiness', 'surprise', 'sadness',
        'anger', 'disgust', 'fear', 'contempt'
    ]
    
    def __init__(self, model_path="models/emotion-ferplus-8.onnx"):
        """
        Initialize emotion predictor
        
        Args:
            model_path: Path to ONNX model file
        """
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            print("✅ Emotion Predictor initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to load emotion model: {e}")
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from face image
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            emotion: Predicted emotion label
            confidence: Confidence score (0-1)
            all_scores: Dictionary of all emotion scores
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size (64x64)
        resized = cv2.resize(gray, (64, 64))
        
        # Normalize and reshape for ONNX model
        # Model expects shape: (1, 1, 64, 64) with values 0-1
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)  # Add channel dimension
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        predictions = outputs[0][0]
        
        # Get emotion with highest score
        emotion_index = np.argmax(predictions)
        confidence = float(predictions[emotion_index])
        emotion = self.EMOTION_LABELS[emotion_index]
        
        # Create dictionary of all emotion scores
        all_scores = {
            label: float(score) 
            for label, score in zip(self.EMOTION_LABELS, predictions)
        }
        
        return emotion, confidence, all_scores
    
    def is_smiling(self, emotion, confidence, threshold=0.5):
        """
        Check if the person is smiling
        
        Args:
            emotion: Predicted emotion
            confidence: Confidence score
            threshold: Minimum confidence threshold
            
        Returns:
            Boolean indicating if person is smiling
        """
        return emotion == 'happiness' and confidence > threshold
