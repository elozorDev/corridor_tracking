import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU kullan
import cv2
from ultralytics import YOLO

print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

print("Opening video file...")
cap = cv2.VideoCapture('sample_video2.mp4')

frame_count = 0
print("\nProcessing video...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO algılama
    results = model(frame, conf=0.5)
    
    # Sonuçları çiz
    annotated_frame = results[0].plot()
    
    # Bilgi yazdır
    detections = results[0].boxes
    human_count = len(detections)
    
    print(f"Frame {frame_count}: {human_count} person(s) detected")
    
    # Ekranda göster
    cv2.imshow('Human Detection', annotated_frame)
    
    # Q tuşu ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Processing complete! Total frames: {frame_count}")