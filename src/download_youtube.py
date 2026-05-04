from pytube import YouTube
import os

# YouTube URL'sini gir
url = "https://www.youtube.com/watch?v=YzcawvDGe4Y"

print("Downloading video...")

try:
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    
    print(f"Video: {yt.title}")
    print(f"Çözünürlük: {stream.resolution}")
    
    stream.download(filename='sample_video.mp4')
    
    print("✅ Video indirildi: sample_video.mp4")
    
except Exception as e:
    print(f"❌ Hata: {e}")