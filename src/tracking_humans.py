import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

print("Opening video file...")
cap = cv2.VideoCapture('sample_video2.mp4')

# Tracking için değişkenler
next_person_id = 1
person_positions = {}  # person_id -> (x, y)
person_history = defaultdict(list)

frame_count = 0
print("\nProcessing video with tracking...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO algılama
    results = model(frame, conf=0.5)
    detections = results[0].boxes
    
    # Yeni pozisyonlar
    current_detections = []
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        current_detections.append((cx, cy, x1, y1, x2, y2))
    
    # Eşleştirme (simple centroid tracking)
    matched_ids = set()
    
    for cx, cy, x1, y1, x2, y2 in current_detections:
        min_distance = float('inf')
        closest_id = None
        
        # En yakın person_id'yi bul
        for pid, (px, py) in person_positions.items():
            distance = np.sqrt((cx - px)**2 + (cy - py)**2)
            if distance < min_distance and distance < 100:  # 100 pixel threshold
                min_distance = distance
                closest_id = pid
        
        if closest_id is not None:
            # Mevcut kişiyi güncelle
            person_positions[closest_id] = (cx, cy)
            person_history[closest_id].append((cx, cy))
            matched_ids.add(closest_id)
        else:
            # Yeni kişi
            person_positions[next_person_id] = (cx, cy)
            person_history[next_person_id].append((cx, cy))
            matched_ids.add(next_person_id)
            next_person_id += 1
        
        # Bounding box çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {closest_id if closest_id else next_person_id-1}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Eski kişileri sil (30 frame görülmemişse)
    for pid in list(person_positions.keys()):
        if pid not in matched_ids:
            del person_positions[pid]
    
    # Bilgi
    person_count = len(current_detections)
    print(f"Frame {frame_count}: {person_count} person(s), Total IDs assigned: {next_person_id-1}")
    
    # Ekranda göster
    cv2.imshow('Human Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Tracking complete! Total frames: {frame_count}, Total unique persons: {next_person_id-1}")