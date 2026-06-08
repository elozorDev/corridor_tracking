# Koridor İzleme (Corridor Tracking)

Koridor kamerasından **kapıları numaralandırıp**, videoda **kişi giriş/çıkış** olaylarını kaydeden sade bir Python uygulaması.

- **YOLOv8** ile insan algılama  
- **Merkez nokta takibi** ile kişi ID’si  
- Her kapı için sanal **dikey çizgi** (`x_position`) geçişi → `ENTER` / `EXIT`  
- JSON rapor: saat, kapı, kişi, özet sayılar  

Repo: [github.com/elozorDev/corridor_tracking](https://github.com/elozorDev/corridor_tracking)

---

## Gereksinimler

- Python **3.10+**
- Windows / Linux / macOS
- **CPU** ile çalışır; NVIDIA GPU + CUDA varsa otomatik hızlanır

GPU kontrolü:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Zorla CPU:

```powershell
python src/corridor_app.py run --video sample_video3.mp4 --device cpu
```

---

## Kurulum

Proje klasörü: `c:\corridor-tracking`

```powershell
cd c:\corridor-tracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

İlk `run` sırasında `yolov8n.pt` modeli indirilir (internet gerekir).

> **Not:** Eski `src/detect_humans.py` gibi dosyalar bilerek CPU’ya kilitlidir. Ana uygulama `corridor_app.py` GPU’yu otomatik kullanır.

---

## Hızlı başlangıç

### 1) Kapıları ayarla (kalibrasyon)

Otomatik renk tabanlı tespit her ortamda doğru olmayabilir. **Önerilen:** elle tıklama.

```powershell
python src/corridor_app.py calibrate --image corridor.jpg --interactive
```

| Tuş | İşlem |
|-----|--------|
| Sol tık | Kapı eşiğine çizgi koy |
| `u` | Son tıkı geri al |
| `q` | Kaydet ve çık |

Çıktılar:

- `config/doors_config.yaml` — kapı listesi ve `x_position`
- `doors_calibrated.jpg` — önizleme (yeşil kutu + turuncu çizgi)

Oda isimlerini YAML’da elle düzenleyebilirsiniz (`name: "5-E"` gibi).

### 2) Videoda izleme

```powershell
python src/corridor_app.py run --video sample_video3.mp4
```

Rapor: `reports/last_run.json`

Canlı kamera / gerçek saat damgası:

```powershell
python src/corridor_app.py run --video 0 --live-clock
```

(`0` = varsayılan webcam; ortamınıza göre değişebilir.)

---

## Komut özeti

```text
python src/corridor_app.py calibrate --image corridor.jpg [--interactive]
python src/corridor_app.py run --video VIDEO.mp4 [--config config/doors_config.yaml]
python src/corridor_app.py run --video VIDEO.mp4 --report reports/rapor.json --conf 0.3 --device cuda
```

---

## Proje yapısı

```text
corridor-tracking/
├── README.md
├── requirements.txt
├── config/
│   ├── doors_config.yaml          # Aktif kapı ayarları (kalibrasyon sonrası)
│   └── doors_config.example.yaml  # Şablon
├── reports/                       # JSON raporlar (git’e eklenmez)
├── corridor.jpg                   # Kalibrasyon görüntüsü (örnek)
├── sample_video3.mp4              # Test videosu (yerelde; büyükse git’e eklenmez)
└── src/
    ├── corridor_app.py            # ★ Ana uygulama
    ├── test_setup.py              # Ortam / GPU testi
    └── ...                        # Eski deneme scriptleri
```

---

## Rapor formatı (`reports/last_run.json`)

```json
{
  "olaylar": [
    {
      "time": "00:00:12.400",
      "person_id": 1,
      "door_name": "Oda 101",
      "action": "ENTER"
    }
  ],
  "ozet_kapi_bazli": {
    "1": {
      "door_name": "Oda 101",
      "giris_olay": 3,
      "cikis_olay": 2,
      "giren_farkli_kisi": 2,
      "cikan_farkli_kisi": 2
    }
  }
}
```

`time`: video başından itibaren süre (`--live-clock` ile gerçek tarih-saat).

---

## Sık sorunlar

| Sorun | Çözüm |
|--------|--------|
| Kapılar yanlış | `--interactive` ile yeniden kalibre et |
| Hiç olay yok | `doors_config.yaml` içindeki `x_position` videodaki kapı hizasında mı kontrol et |
| Yavaş | `--device cuda` veya daha küçük model |
| `python` bulunamıyor | Python 3 kur, PATH’e ekle, sanal ortamı aktive et |

---

## GitHub’a gönderme

```powershell
cd c:\corridor-tracking
git status
git push origin main
```

Yeni dosyalar için (örnek):

```powershell
git add README.md .gitignore config/ reports/.gitkeep corridor.jpg
git add src/corridor_app.py
git commit -m "Add README, gitignore and corridor tracking setup"
git push origin main
```

`.gitignore` sanal ortam, büyük videolar ve model dosyasını hariç tutar.

---

## Lisans

Bu depo eğitim / proje amaçlıdır. Kendi kurum politikanıza göre kullanın.
