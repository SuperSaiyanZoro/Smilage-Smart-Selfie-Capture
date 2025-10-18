import cv2
from utils.face_detector import FaceDetector

def test_face_detection():
    """Test face detection with webcam"""
    
    print("="*60)
    print("🎥 Testing Face Detection with Webcam")
    print("="*60)
    print("\nInstructions:")
    print("  - Position your face in front of the camera")
    print("  - Green rectangle should appear around your face")
    print("  - Press 'q' to quit\n")
    
    # Initialize face detector
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("📸 Starting face detection... (Press 'q' to quit)\n")
    
    frame_count = 0
    faces_detected = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if len(faces) > 0:
            faces_detected += 1
        
        # Draw rectangles around faces
        frame = detector.draw_faces(frame, faces)
        
        # Add info text
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow('Face Detection Test - Press Q to quit', frame)
        
        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 Detection Statistics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames with faces: {faces_detected}")
    print(f"  Detection rate: {(faces_detected/frame_count)*100:.1f}%")
    print("="*60)
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_face_detection()
