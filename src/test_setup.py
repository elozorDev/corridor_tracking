import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("=" * 50)
print("CORRIDOR TRACKING SYSTEM - SETUP TEST")
print("=" * 50)

print("\n1. OpenCV:")
print(f"   Version: {cv2.__version__}")

print("\n2. NumPy:")
print(f"   Version: {np.__version__}")

print("\n3. PyTorch:")
print(f"   Version: {torch.__version__}")
print(f"   GPU Available: {torch.cuda.is_available()}")

print("\n4. YOLOv8 Model (loading...):")
try:
    model = YOLO('yolov8n.pt')
    print("   ✅ YOLOv8 loaded successfully!")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("✅ SETUP SUCCESSFUL!")
print("=" * 50)