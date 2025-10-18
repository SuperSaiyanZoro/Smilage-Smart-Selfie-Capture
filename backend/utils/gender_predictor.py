import cv2
import numpy as np

class GenderPredictor:
    """
    Gender prediction using pre-trained Caffe model
    """
    
    # Gender labels
    GENDER_LIST = ['Male', 'Female']
    
    def __init__(
        self,
        prototxt_path="models/gender_deploy.prototxt",
        model_path="models/gender_net.caffemodel"
    ):
        """
        Initialize gender predictor
        
        Args:
            prototxt_path: Path to model architecture file
            model_path: Path to model weights file
        """
        try:
            self.gender_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("✅ Gender Predictor initialized successfully")
        except Exception as e:
            raise Exception(f"Failed to load gender model: {e}")
    
    def predict_gender(self, face_image):
        """
        Predict gender from face image
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            gender: Predicted gender ('Male' or 'Female')
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
        self.gender_net.setInput(blob)
        
        # Get predictions
        predictions = self.gender_net.forward()
        
        # Get the gender with highest confidence
        gender_index = predictions[0].argmax()
        confidence = predictions[0][gender_index]
        gender = self.GENDER_LIST[gender_index]
        
        return gender, float(confidence)
