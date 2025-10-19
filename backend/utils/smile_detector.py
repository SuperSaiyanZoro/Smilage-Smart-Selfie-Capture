import cv2

class SmileDetector:
    """
    Smile detection using OpenCV Haar Cascade
    This is more reliable than the emotion model for smile detection
    """
    
    def __init__(self, cascade_path="models/haarcascade_smile.xml"):
        """
        Initialize smile detector
        
        Args:
            cascade_path: Path to Haar Cascade XML file
        """
        self.smile_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.smile_cascade.empty():
            raise Exception(f"Failed to load smile cascade from {cascade_path}")
        
        print("✅ Smile Detector initialized successfully")
    
    def detect_smile(self, face_image, min_neighbors=20):
        """
        Detect smile in face image
        
        Args:
            face_image: Face image (BGR or grayscale)
            min_neighbors: Detection sensitivity (higher = stricter)
            
        Returns:
            is_smiling: Boolean
            confidence: Number of smile detections (higher = more confident)
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Detect smiles
        smiles = self.smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=min_neighbors,
            minSize=(25, 25)
        )
        
        # Check if smile detected
        is_smiling = len(smiles) > 0
        confidence = len(smiles)  # More detections = more confident
        
        return is_smiling, confidence, smiles
    
    def get_smile_score(self, face_image):
        """
        Get normalized smile score (0.0 to 1.0)
        
        Args:
            face_image: Face image
            
        Returns:
            smile_score: Float between 0.0 (not smiling) and 1.0 (smiling)
        """
        is_smiling, confidence, _ = self.detect_smile(face_image, min_neighbors=15)
        
        # Normalize confidence to 0.0-1.0 range
        # Typically 0-5 smile detections
        smile_score = min(confidence / 5.0, 1.0)
        
        return smile_score
