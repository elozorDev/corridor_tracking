"""
Koridor izleme: resimden kapı kalibrasyonu + videoda giriş/çıkış sayımı.
Kullanım (proje kökünden):
  python src/corridor_app.py calibrate --image corridor.jpg
  python src/corridor_app.py calibrate --image corridor.jpg --interactive
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


def _bbox_dict(x: int, y: int, w: int, h: int) -> dict:
    return {"x": x, "y": y, "w": w, "h": h, "cx": x + w // 2, "cy": y + h // 2}


def _inter_area(a: dict, b: dict) -> int:
    xa, ya, wa, ha = a["x"], a["y"], a["w"], a["h"]
    xb, yb, wb, hb = b["x"], b["y"], b["w"], b["h"]
    ix1, iy1 = max(xa, xb), max(ya, yb)
    ix2, iy2 = min(xa + wa, xb + wb), min(ya + ha, yb + hb)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


def _iou(a: dict, b: dict) -> float:
    inter = _inter_area(a, b)
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def _merge_union(a: dict, b: dict) -> dict:
    x1 = min(a["x"], b["x"])
    y1 = min(a["y"], b["y"])
    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
    y2 = max(a["y"] + a["h"], b["y"] + b["h"])
    w, h = x2 - x1, y2 - y1
    return _bbox_dict(x1, y1, w, h)


def _cluster_door_boxes(boxes: list[dict], iou_thresh: float = 0.22, near_px: int = 55) -> list[dict]:
    """Aynı kapıya binen / çok yakın kutuları tek kutuda birleştirir (tekrarlı)."""
    boxes = [dict(b) for b in boxes]
    while len(boxes) > 1:
        merged_any = False
        new_list: list[dict] = []
        for b in sorted(boxes, key=lambda x: x["w"] * x["h"], reverse=True):
            hit = False
            for i, k in enumerate(new_list):
                iou = _iou(b, k)
                sm = min(b["w"] * b["h"], k["w"] * k["h"])
                inter_ratio = _inter_area(b, k) / sm if sm > 0 else 0.0
                near = abs(b["cx"] - k["cx"]) < near_px and _vertical_overlap_ratio(b, k) > 0.35
                if iou >= iou_thresh or inter_ratio >= 0.45 or near:
                    new_list[i] = _merge_union(b, k)
                    merged_any = True
                    hit = True
                    break
            if not hit:
                new_list.append(dict(b))
        if not merged_any:
            break
        boxes = new_list
    return boxes


def _vertical_overlap_ratio(a: dict, b: dict) -> float:
    ya, yb = max(a["y"], b["y"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter_h = max(0, yb - ya)
    mh = min(a["h"], b["h"])
    return inter_h / mh if mh > 0 else 0.0


def _reject_likely_stairs(img_h: int, img_w: int, b: dict) -> bool:
    """Sol-alt merdiven / zemin gürültüsü (heuristik)."""
    cx, cy = b["cx"], b["cy"]
    bottom = b["y"] + b["h"]
    if cx < 0.18 * img_w and bottom > 0.88 * img_h and cy > 0.62 * img_h:
        return True
    if cx < 0.10 * img_w and bottom > 0.82 * img_h:
        return True
    return False


def _auto_door_boxes(img: np.ndarray) -> list[dict]:
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 150, 150]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 2000 and h > w and h > 80:
            raw.append(_bbox_dict(x, y, w, h))

    raw = [b for b in raw if not _reject_likely_stairs(H, W, b)]
    merged = _cluster_door_boxes(raw)
    merged.sort(key=lambda b: b["cx"])
    return merged


def calibrate(
    image_path: Path,
    config_out: Path,
    preview_out: Path | None,
    *,
    interactive: bool = False,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Resim okunamadı: {image_path}")

    if interactive:
        candidates = _calibrate_interactive_clicks(img)
    else:
        candidates = _auto_door_boxes(img)

    if not candidates:
        raise SystemExit(
            "Hiç kapı kalmadı. Sol-alt merdiven filtrelendi veya renk eşiği uyumsuz. "
            "Tekrar: python src/corridor_app.py calibrate --image corridor.jpg --interactive"
        )
    doors = [
        {"id": i + 1, "name": f"Oda {i + 1}", "x_position": int(b["cx"])}
        for i, b in enumerate(candidates)
    ]

    config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"doors": doors}, f, allow_unicode=True, sort_keys=False)

    vis = img.copy()
    H, W = vis.shape[:2]
    for d, b in zip(doors, candidates):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(vis, str(d["id"]), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        xv = int(d["x_position"])
        cv2.line(vis, (xv, 0), (xv, H - 1), (0, 180, 255), 2)

    if preview_out:
        cv2.imwrite(str(preview_out), vis)
    print(f"{len(doors)} kapı yazıldı: {config_out}")
    if preview_out:
        print(f"Önizleme: {preview_out}")


def _calibrate_interactive_clicks(img: np.ndarray) -> list[dict]:
    """Sol tık: kapı çizgisi (soldan sağa numara otomatik sıralanır). u: geri al, q: kaydet."""
    H, W = img.shape[:2]
    max_w = 1400
    scale = min(1.0, max_w / float(W))
    disp_w, disp_h = int(W * scale), int(H * scale)

    clicks: list[tuple[int, int]] = []

    def to_orig(xd: int, yd: int) -> tuple[int, int]:
        return int(xd / scale), int(yd / scale)

    def on_mouse(event: int, xd: int, yd: int, _f: int, _p: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            xo, yo = to_orig(xd, yd)
            clicks.append((xo, yo))

    win = "Kalibrasyon — sol tik kapı | u geri | q bitir"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)

    print("Pencerede her kapının eşiğinde/çerçevesinde bir kez SOL TIK yap (koridor boyunca).")
    print("'u' son tıkı siler, 'q' YAML kaydeder ve çıkar.")

    while True:
        frame = img.copy()
        for i, (xc, yc) in enumerate(clicks):
            cv2.line(frame, (xc, 0), (xc, H - 1), (0, 200, 255), 2)
            cv2.putText(frame, str(i + 1), (xc + 6, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
        view = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        if k == ord("u") and clicks:
            clicks.pop()

    cv2.destroyWindow(win)

    if not clicks:
        raise SystemExit("Hiç tıklanmadı; en az bir kapı işaretleyin.")

    clicks_sorted = sorted(clicks, key=lambda t: t[0])
    out: list[dict] = []
    for xc, yc in clicks_sorted:
        x0 = max(0, xc - 4)
        out.append(_bbox_dict(x0, 0, 8, H))
    return out


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
    c.add_argument(
        "--interactive",
        action="store_true",
        help="Otomatik yerine: resimde her kapıya tıkla (en doğru yöntem)",
    )

    r = sub.add_parser("run", help="Videoda kişi takibi ve kapı geçişi")
    r.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    r.add_argument("--report", type=Path, default=ROOT / "reports" / "last_run.json")
    r.add_argument("--live-clock", action="store_true", help="Gerçek saat damgası (webcam/canlı için)")
    r.add_argument("--conf", type=float, default=0.35)

    args = p.parse_args()
    if args.cmd == "calibrate":
        calibrate(args.image, args.config, args.preview, interactive=args.interactive)
    else:
        run_monitor(args.video, args.config, args.report, live_clock=args.live_clock, conf=args.conf)


if __name__ == "__main__":
    main()
