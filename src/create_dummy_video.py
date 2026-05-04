import cv2
import numpy as np

# Video yazıcı oluştur
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (1280, 720))

print("Creating test video (5 seconds)...")

# 5 saniye video (30 FPS = 150 frame)
for i in range(150):
    # Siyah frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Hareketli yeşil dikdörtgen (insan simulasyonu)
    x = int(100 + i * 4)
    y = 300
    cv2.rectangle(frame, (x, y), (x+50, y+100), (0, 255, 0), -1)
    
    # Yazı
    cv2.putText(frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Test Video", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Frame'i yaz
    out.write(frame)
    
    if i % 30 == 0:
        print(f"  Frame {i}/150")

out.release()
print("✅ test_video.mp4 created successfully!")