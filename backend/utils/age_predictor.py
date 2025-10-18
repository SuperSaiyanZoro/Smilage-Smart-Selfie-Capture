import cv2
import numpy as np

class AgePredictor:
    """
    Age prediction using pre-trained Caffe model
    """
    
    # Age ranges that the model predicts
    AGE_RANGES = [
        '(0-2)', '(4-6)', '(8-12)', '(15-20)',
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'
    ]
    
    def __init__(
        self,
        prototxt_path="models/age_deploy.prototxt",
        model_path="models/age_net.caffemodel"
    ):
        """
        Initialize age predictor
        
        Args:
            prototxt_path: Path to model architecture file
            model_path: Path to model weights file
        """
        try:
            self.age_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("✅ Age Predictor initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to load age model: {e}")
    
    def predict_age(self, face_image):
        """
        Predict age from face image
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            age_range: Predicted age range as string
            confidence: Confidence score (0-1)
        """
        # Prepare the face image for the model
        blob = cv2.dnn.blobFromImage(
            face_image,
            scalefactor=1.0,
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )
        
        # Feed the image to the network
        self.age_net.setInput(blob)
        
        # Get predictions
        predictions = self.age_net.forward()
        
        # Get the age range with highest confidence
        age_index = predictions[0].argmax()
        confidence = predictions[0][age_index]
        age_range = self.AGE_RANGES[age_index]
        
        return age_range, float(confidence)
    
    def get_age_midpoint(self, age_range):
        """
        Convert age range to midpoint value
        
        Args:
            age_range: Age range string like "(25-32)"
            
        Returns:
            Midpoint age value
        """
        # Remove parentheses and split
        ages = age_range.strip('()').split('-')
        low = int(ages[0])
        high = int(ages[1])
        return (low + high) // 2
