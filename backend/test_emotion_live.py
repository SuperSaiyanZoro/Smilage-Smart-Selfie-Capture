import cv2
from utils.emotion_predictor import EmotionPredictor

predictor = EmotionPredictor()
cap = cv2.VideoCapture(0)

print("Testing live emotion detection...")
print("Try different expressions and watch the scores change\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Get center crop as "face"
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2
    size = min(h, w) // 2
    face = frame[cy-size:cy+size, cx-size:cx+size]
    
    if face.shape[0] > 0:
        emotion, conf, all_emotions = predictor.predict_emotion(face)
        
        # Print every 10 frames
        if frame_count % 10 == 0:
            print(f"\nFrame {frame_count}:")
            print(f"  Top emotion: {emotion} ({conf:.3f})")
            print(f"  Happiness: {all_emotions['happiness']:.3f}")
            print(f"  Neutral: {all_emotions['neutral']:.3f}")
    
    cv2.imshow('Test (press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
