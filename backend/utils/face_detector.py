import cv2
import numpy as np

class FaceDetector:
    """
    Face detection using OpenCV Haar Cascades
    """
    
    def __init__(self, cascade_path="models/haarcascade_frontalface_default.xml"):
        """
        Initialize face detector with Haar Cascade
        
        Args:
            cascade_path: Path to Haar Cascade XML file
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception(f"Failed to load cascade classifier from {cascade_path}")
        
        print("✅ Face Detector initialized successfully")
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of face rectangles (x, y, w, h)
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def draw_faces(self, frame, faces):
        """
        Draw rectangles around detected faces
        
        Args:
            frame: Input image
            faces: List of face rectangles
            
        Returns:
            Frame with drawn rectangles
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
    
    def extract_face(self, frame, face_rect):
        """
        Extract face region from frame
        
        Args:
            frame: Input image
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            Extracted face image
        """
        x, y, w, h = face_rect
        face = frame[y:y+h, x:x+w]
        return face
    
    def check_blur(self, face_image, threshold=100):
        """
        Check if face image is blurry using Laplacian variance
        
        Args:
            face_image: Face image
            threshold: Blur threshold (lower = more blurry)
            
        Returns:
            is_clear: Boolean indicating if image is clear
            blur_score: Numeric blur score
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_clear = blur_score > threshold
        
        return is_clear, blur_score
