"""
Koridor izleme: resimden kapı kalibrasyonu + videoda giriş/çıkış sayımı.
Kullanım (proje kökünden):
  python src/corridor_app.py calibrate --image corridor.jpg
  python src/corridor_app.py run --video sample_video3.mp4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "doors_config.yaml"
DEFAULT_CALIB_IMAGE = ROOT / "corridor.jpg"
DEFAULT_VIDEO = ROOT / "sample_video3.mp4"


def load_config(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["doors"]


def calibrate(image_path: Path, config_out: Path, preview_out: Path | None) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Resim okunamadı: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 150, 150]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 2000 and h > w and h > 80:
            cx = x + w // 2
            candidates.append({"cx": cx, "x": x, "y": y, "w": w, "h": h})

    candidates.sort(key=lambda d: d["cx"])
    doors = [
        {"id": i + 1, "name": f"Oda {i + 1}", "x_position": int(b["cx"])} for i, b in enumerate(candidates)
    ]

    config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"doors": doors}, f, allow_unicode=True, sort_keys=False)

    vis = img.copy()
    for d, b in zip(doors, candidates):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(vis, str(d["id"]), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    if preview_out:
        cv2.imwrite(str(preview_out), vis)
    print(f"{len(doors)} kapı yazıldı: {config_out}")
    if preview_out:
        print(f"Önizleme: {preview_out}")


@dataclass
class MonitorState:
    next_id: int = 1
    people: dict[int, tuple[int, int, int]] = field(default_factory=dict)  # id -> (cx, cy, last_frame)
    events: list[dict] = field(default_factory=list)


def _match_centroid(
    cx: int, cy: int, people: dict[int, tuple[int, int, int]], used: set[int], max_dist: float
) -> int | None:
    best_id, best_d = None, max_dist
    for pid, (px, py, _) in people.items():
        if pid in used:
            continue
        d = float(np.hypot(cx - px, cy - py))
        if d < best_d:
            best_d, best_id = d, pid
    return best_id


def run_monitor(
    video_path: Path,
    config_path: Path,
    report_path: Path | None,
    model_name: str = "yolov8n.pt",
    match_px: int = 55,
    lost_frames: int = 100,
    conf: float = 0.35,
    live_clock: bool = False,
) -> None:
    doors = load_config(config_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    model = YOLO(model_name)
    st = MonitorState()
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        if live_clock:
            t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            sec = (frame_i - 1) / fps
            whole = int(sec)
            ms = int(round((sec - whole) * 1000))
            h, r = divmod(whole, 3600)
            m, s = divmod(r, 60)
            t_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        dets = model(frame, conf=conf, verbose=False)[0].boxes
        centers: list[tuple[int, int]] = []
        for b in dets:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

        matched: set[int] = set()
        used_ids: set[int] = set()
        for cx, cy in centers:
            pid = _match_centroid(cx, cy, st.people, used_ids, float(match_px))
            if pid is not None:
                used_ids.add(pid)
                old_x, _, _ = st.people[pid]
                st.people[pid] = (cx, cy, frame_i)
                matched.add(pid)

                for d in doors:
                    dx = d["x_position"]
                    if old_x < dx <= cx:
                        st.events.append(
                            {
                                "frame": frame_i,
                                "time": t_str,
                                "person_id": pid,
                                "door_id": d["id"],
                                "door_name": d["name"],
                                "action": "ENTER",
                            }
                        )
                    elif old_x > dx >= cx:
                        st.events.append(
                            {
                                "frame": frame_i,
                                "time": t_str,
                                "person_id": pid,
                                "door_id": d["id"],
                                "door_name": d["name"],
                                "action": "EXIT",
                            }
                        )
            else:
                st.people[st.next_id] = (cx, cy, frame_i)
                matched.add(st.next_id)
                st.next_id += 1

        for pid in list(st.people.keys()):
            if pid not in matched and frame_i - st.people[pid][2] > lost_frames:
                del st.people[pid]

    cap.release()

    by_door: dict[int, dict] = defaultdict(
        lambda: {
            "enter_events": 0,
            "exit_events": 0,
            "enter_persons": set(),
            "exit_persons": set(),
        }
    )
    for e in st.events:
        dd = by_door[e["door_id"]]
        if e["action"] == "ENTER":
            dd["enter_events"] += 1
            dd["enter_persons"].add(e["person_id"])
        else:
            dd["exit_events"] += 1
            dd["exit_persons"].add(e["person_id"])

    summary = {}
    for did, dd in by_door.items():
        summary[did] = {
            "door_name": next(d["name"] for d in doors if d["id"] == did),
            "giris_olay": dd["enter_events"],
            "cikis_olay": dd["exit_events"],
            "giren_farkli_kisi": len(dd["enter_persons"]),
            "cikan_farkli_kisi": len(dd["exit_persons"]),
        }

    out = {
        "video": str(video_path),
        "frames": frame_i,
        "fps": fps,
        "zaman_kaynagi": "canli_saat" if live_clock else "video_offset",
        "olaylar": st.events,
        "ozet_kapi_bazli": summary,
    }

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        serial = json.loads(json.dumps(out, default=str))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)
        print(f"Rapor: {report_path}")

    print("\n--- Olaylar ---")
    for e in st.events:
        print(f"{e['time']} | kişi {e['person_id']} | {e['door_name']} | {e['action']}")

    print("\n--- Özet (kapı bazlı) ---")
    for did in sorted(summary.keys()):
        s = summary[did]
        print(
            f"{s['door_name']}: giriş olayı={s['giris_olay']}, çıkış olayı={s['cikis_olay']}, "
            f"farklı giren={s['giren_farkli_kisi']}, farklı çıkan={s['cikan_farkli_kisi']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Koridor kapı kalibrasyonu ve giriş/çıkış izleme")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="Koridor görüntüsünden kapıları bul ve YAML yaz")
    c.add_argument("--image", type=Path, default=DEFAULT_CALIB_IMAGE)
    c.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    c.add_argument("--preview", type=Path, default=ROOT / "doors_calibrated.jpg")

    r = sub.add_parser("run", help="Videoda kişi takibi ve kapı geçişi")
    r.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    r.add_argument("--report", type=Path, default=ROOT / "reports" / "last_run.json")
    r.add_argument("--live-clock", action="store_true", help="Gerçek saat damgası (webcam/canlı için)")
    r.add_argument("--conf", type=float, default=0.35)

    args = p.parse_args()
    if args.cmd == "calibrate":
        calibrate(args.image, args.config, args.preview)
    else:
        run_monitor(args.video, args.config, args.report, live_clock=args.live_clock, conf=args.conf)


if __name__ == "__main__":
    main()
