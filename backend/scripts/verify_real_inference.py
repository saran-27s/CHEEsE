"""Run the REAL reference models on public validation images and report measured numbers.

Inputs (downloaded on demand into backend/verification/, git-ignored, CC BY 4.0, Roboflow
workspace chess-detection-8s15a via Hugging Face datasets by surawut):
  * surawut/chess-pieces-dataset-yolo  valid/  -> warped 640x640 boards with YOLO box labels
  * surawut/chessboard-dataset-yolo    valid/  -> raw frames with 4 corner keypoints

Measured:
  * piece detector: box precision/recall at IoU >= 0.5 (same class), and square-level agreement
    between the board built from predictions and the board built from labels (same anchor rule)
  * corner model: mean / max pixel error of suggested a1/h1/a8/h8 corners vs labels
  * CPU latency per image

Usage: python scripts/verify_real_inference.py [--pieces 24] [--corners 12]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chessvision.config import load_settings  # noqa: E402
from chessvision.core.observation import AnchorSpec, Detection, build_observation  # noqa: E402
from chessvision.inference.registry import ModelRegistry  # noqa: E402

OUT = ROOT / "verification"
PIECES = "surawut/chess-pieces-dataset-yolo"
CORNERS = "surawut/chessboard-dataset-yolo"
PIECE_NAMES = ['Hand', 'bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']  # data.yaml order


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "chess-vision-verify"}), timeout=60) as r:
        return json.loads(r.read())


def fetch(repo: str, path: str) -> Path:
    dest = OUT / repo.replace("/", "__") / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "chess-vision-verify"}), timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def pick(repo: str, n: int) -> list[tuple[str, str]]:
    files = [s["rfilename"] for s in get_json(f"https://huggingface.co/api/datasets/{repo}")["siblings"]]
    images = sorted(f for f in files if f.startswith("valid/images/"))
    labels = set(files)
    pairs = []
    for img in images:
        lbl = "valid/labels/" + Path(img).with_suffix(".txt").name
        if lbl in labels:
            pairs.append((img, lbl))
    if not pairs:
        raise SystemExit(f"no validation image/label pairs found in {repo}")
    step = max(1, len(pairs) // n)
    return pairs[::step][:n]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def verify_pieces(registry: ModelRegistry, n: int, settings) -> dict:
    det = registry.piece_detector()
    model_names = list(det.names.values())
    pairs = pick(PIECES, n)
    class_map = det.spec.class_map
    tp = fp = fn = 0
    square_agree, exact_boards, latencies, per_image = [], 0, [], []
    anchor = AnchorSpec(mode="toward_camera")  # warped dataset images: no camera direction -> box centre
    for img_path, lbl_path in pairs:
        image = cv2.imread(str(fetch(PIECES, img_path)))
        h, w = image.shape[:2]
        warped = cv2.resize(image, (settings.warp_size, settings.warp_size)) if (h, w) != (settings.warp_size,) * 2 else image
        sx, sy = settings.warp_size / w, settings.warp_size / h
        gt = []
        for line in fetch(PIECES, lbl_path).read_text().split("\n"):
            parts = line.split()
            if len(parts) < 5:
                continue
            k, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            S = settings.warp_size
            gt.append(Detection(label=class_map[PIECE_NAMES[k]], conf=1.0,
                                box=((cx - bw / 2) * S, (cy - bh / 2) * S, (cx + bw / 2) * S, (cy + bh / 2) * S)))
        t0 = time.perf_counter()
        pred = det.detect(warped)
        latencies.append(time.perf_counter() - t0)
        used = set()
        img_tp = 0
        for p in sorted(pred, key=lambda d: -d.conf):
            best, best_i = 0.0, None
            for i, g in enumerate(gt):
                if i in used or g.label != p.label:
                    continue
                v = iou(p.box, g.box)
                if v > best:
                    best, best_i = v, i
            if best_i is not None and best >= 0.5:
                used.add(best_i)
                img_tp += 1
        tp += img_tp
        fp += len(pred) - img_tp
        fn += len(gt) - img_tp
        kw = dict(anchor=anchor, camera_dir=None, empty_prior=settings.observation.empty_prior)
        ob_pred = build_observation(pred, settings.warp_size, **kw)
        ob_gt = build_observation(gt, settings.warp_size, **kw)
        known = [i for i in range(64) if not ob_gt.squares[i].occluded and not ob_pred.squares[i].occluded]
        agree = sum(1 for i in known if ob_pred.squares[i].top == ob_gt.squares[i].top)
        frac = agree / len(known) if known else float("nan")
        square_agree.append(frac)
        exact_boards += int(bool(known) and agree == len(known))
        per_image.append({"image": img_path, "gt_boxes": len(gt), "pred_boxes": len(pred), "tp": img_tp,
                          "squares_compared": len(known), "squares_agree": agree,
                          "hand_in_labels": any(g.label == "hand" for g in gt)})
    return {
        "model_id": det.spec.id, "model_class_names": model_names,
        "class_names_match_data_yaml": sorted(model_names) == sorted(PIECE_NAMES),
        "images": len(pairs), "device": det.device,
        "box_precision_iou50": tp / (tp + fp) if tp + fp else None,
        "box_recall_iou50": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
        "square_agreement_mean": statistics.mean(x for x in square_agree if x == x),
        "square_agreement_min": min(x for x in square_agree if x == x),
        "boards_fully_agreeing": exact_boards,
        "latency_s_mean": statistics.mean(latencies), "latency_s_max": max(latencies),
        "latency_note": "first call includes model warm-up",
        "per_image": per_image,
    }


def verify_corners(registry: ModelRegistry, n: int) -> dict:
    det = registry.corner_detector()
    pairs = pick(CORNERS, n)
    errors, misses, latencies, rows = [], 0, [], []
    for img_path, lbl_path in pairs:
        image = cv2.imread(str(fetch(CORNERS, img_path)))
        h, w = image.shape[:2]
        parts = fetch(CORNERS, lbl_path).read_text().split()
        kp = np.array(list(map(float, parts[5:5 + 12]))).reshape(4, 3)
        gt = {name: (kp[i][0] * w, kp[i][1] * h) for i, name in enumerate(["a1", "h1", "a8", "h8"])}
        t0 = time.perf_counter()
        s = det.suggest(image)
        latencies.append(time.perf_counter() - t0)
        if s is None:
            misses += 1
            rows.append({"image": img_path, "found": False})
            continue
        ring = s.corners[s.a1_index:] + s.corners[:s.a1_index]  # a1, a8, h8, h1
        pred = dict(zip(["a1", "a8", "h8", "h1"], ring))
        e = [float(np.hypot(pred[k][0] - gt[k][0], pred[k][1] - gt[k][1])) for k in gt]
        errors.extend(e)
        rows.append({"image": img_path, "found": True, "max_px_error": max(e), "image_size": [w, h],
                     "confidence": s.confidence})
    return {"model_id": det.spec.id, "images": len(pairs), "not_found": misses,
            "corner_px_error_mean": statistics.mean(errors) if errors else None,
            "corner_px_error_median": statistics.median(errors) if errors else None,
            "corner_px_error_max": max(errors) if errors else None,
            "latency_s_mean": statistics.mean(latencies), "per_image": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=24)
    ap.add_argument("--corners", type=int, default=12)
    args = ap.parse_args()
    settings = load_settings()
    registry = ModelRegistry(settings.models_manifest)
    status = registry.status()
    print(json.dumps([{k: s[k] for k in ("id", "status", "message")} for s in status], indent=1))
    OUT.mkdir(exist_ok=True)
    results = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model_status": status,
               "device": registry.device(),
               "pieces": verify_pieces(registry, args.pieces, settings),
               "corners": verify_corners(registry, args.corners)}
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    p, c = results["pieces"], results["corners"]
    print(f"pieces: {p['images']} images, precision {p['box_precision_iou50']:.3f}, recall {p['box_recall_iou50']:.3f}, "
          f"square agreement mean {p['square_agreement_mean']:.3f} (min {p['square_agreement_min']:.3f}), "
          f"boards fully agreeing {p['boards_fully_agreeing']}/{p['images']}, "
          f"latency mean {p['latency_s_mean']:.3f}s on {p['device']}; class names match data.yaml: "
          f"{p['class_names_match_data_yaml']}")
    print(f"corners: {c['images']} images, not found {c['not_found']}, px error mean {c['corner_px_error_mean']}, "
          f"median {c['corner_px_error_median']}, max {c['corner_px_error_max']}, latency mean {c['latency_s_mean']:.3f}s")
    print(f"full results: {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
