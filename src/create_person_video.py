import cv2
import numpy as np

# Video yazıcı
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sample_video.mp4', fourcc, 30.0, (1280, 720))

print("Creating video with moving rectangles (simulating people)...")

for i in range(150):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # İnsan 1 (mavi dikdörtgen)
    x1 = int(100 + i * 3)
    cv2.rectangle(frame, (x1, 200), (x1+80, 400), (255, 0, 0), -1)  # Mavi
    
    # İnsan 2 (kırmızı dikdörtgen)
    x2 = int(50 + i * 2)
    cv2.rectangle(frame, (x2, 400), (x2+80, 600), (0, 0, 255), -1)  # Kırmızı
    
    # İnsan 3 (yeşil dikdörtgen)
    x3 = int(200 + i * 4)
    cv2.rectangle(frame, (x3, 100), (x3+80, 300), (0, 255, 0), -1)  # Yeşil
    
    cv2.putText(frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)
    
    if i % 30 == 0:
        print(f"  Frame {i}/150")

out.release()
print("✅ sample_video.mp4 created with moving objects!")