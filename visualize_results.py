#!/usr/bin/env python3
"""
Visualize segmentation results từ submissions/.
Lưu ảnh overlay vào viz/ để xem trong VSCode.

Chạy:
  python3 visualize_results.py              # tất cả task
  python3 visualize_results.py easy_semantic
  python3 visualize_results.py medium_instance
"""
import io, json, sys, zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

ROOT        = Path(__file__).parent
DATA        = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
VIZ         = ROOT / "viz"
VIZ.mkdir(exist_ok=True)

ALPHA = 0.45   # mask opacity

# Màu ngẫu nhiên dùng cho instance (mỗi instance 1 màu)
RNG = np.random.default_rng(42)

def random_color():
    return tuple(int(c) for c in RNG.integers(60, 230, 3))

# ---------------------------------------------------------------------------
# Semantic: color PNG mask → overlay lên ảnh gốc
# ---------------------------------------------------------------------------
def viz_semantic(task_name):
    zip_path = SUBMISSIONS / f"{task_name}.zip"
    if not zip_path.exists():
        print(f"[skip] {task_name}.zip không tìm thấy"); return

    # Tìm thư mục ảnh gốc
    info = json.loads((DATA / "manifest.json").read_text())["tasks"][task_name]
    img_dir = DATA / info["path"] / "images"

    out_dir = VIZ / task_name
    out_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        pngs = {Path(n).stem: n for n in z.namelist() if "SegmentationClass/" in n and n.endswith(".png")}
        lm_key = next(k for k in z.namelist() if k.endswith("labelmap.txt"))
        # Parse labelmap
        color2label = {}
        for line in z.read(lm_key).decode().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            parts = line.split(":")
            name = parts[0]
            if not parts[1]:
                continue
            try:
                r, g, b = map(int, parts[1].split(","))
                color2label[(r, g, b)] = name
            except ValueError:
                continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            if stem not in pngs:
                print(f"  [warn] mask không có cho {img_path.name}"); continue

            orig = np.array(Image.open(img_path).convert("RGB"))
            mask_rgb = np.array(Image.open(io.BytesIO(z.read(pngs[stem]))).convert("RGB"))

            # Overlay: blend mask màu lên ảnh gốc
            overlay = orig.copy().astype(np.float32)
            colored = np.zeros_like(orig, dtype=np.float32)
            has_label = np.zeros(orig.shape[:2], bool)

            for (r, g, b), label in color2label.items():
                if label == "background":
                    continue
                px = (mask_rgb[:,:,0]==r) & (mask_rgb[:,:,1]==g) & (mask_rgb[:,:,2]==b)
                if px.any():
                    colored[px] = [r, g, b]
                    has_label |= px

            overlay[has_label] = (
                orig[has_label].astype(np.float32) * (1 - ALPHA) +
                colored[has_label] * ALPHA
            )
            result = overlay.astype(np.uint8)

            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            out_path = out_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_path), result_bgr)
            print(f"  saved → viz/{task_name}/{stem}.jpg")

# ---------------------------------------------------------------------------
# Instance / Panoptic: COCO polygon → vẽ màu từng instance
# ---------------------------------------------------------------------------
def viz_instance(task_name):
    zip_path = SUBMISSIONS / f"{task_name}.zip"
    if not zip_path.exists():
        print(f"[skip] {task_name}.zip không tìm thấy"); return

    info = json.loads((DATA / "manifest.json").read_text())["tasks"][task_name]
    img_dir = DATA / info["path"] / "images"

    out_dir = VIZ / task_name
    out_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        json_key = next(k for k in z.namelist() if k.endswith(".json") and "annotation" in k.lower())
        coco = json.loads(z.read(json_key))

    cats  = {c["id"]: c["name"] for c in coco["categories"]}
    imgs  = {im["id"]: im for im in coco["images"]}

    # Màu cố định per category
    cat_colors = {}
    palette = [
        (220,20,60),(0,0,142),(119,11,32),(0,0,230),
        (0,60,100),(0,0,70),(250,170,30),(107,142,35),
        (70,130,180),(70,70,70),(244,35,232),(128,64,128),
    ]
    for i, (cid, cname) in enumerate(sorted(cats.items())):
        cat_colors[cid] = palette[i % len(palette)]

    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"  [warn] ảnh gốc không có: {img_path}"); continue

        orig_bgr = cv2.imread(str(img_path))
        H, W = orig_bgr.shape[:2]
        overlay = orig_bgr.copy()

        anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]

        # Rasterize polygon hoặc RLE (giống scorer), mask lớn vẽ trước, nhỏ đè lên sau
        def raster(ann):
            seg = ann["segmentation"]
            if isinstance(seg, list):
                r = cocomask.merge(cocomask.frPyObjects(seg, H, W))
            elif isinstance(seg.get("counts"), list):
                r = cocomask.frPyObjects(seg, H, W)
            else:
                r = seg
            return cocomask.decode(r).astype(bool)
        masks = sorted(((a, raster(a)) for a in anns), key=lambda t: -int(t[1].sum()))
        for ann, m in masks:
            color = cat_colors.get(ann["category_id"], (128,128,128))
            overlay[m] = (color[2], color[1], color[0])

        # Blend
        result = cv2.addWeighted(orig_bgr, 1 - ALPHA, overlay, ALPHA, 0)

        # Vẽ contour + label text
        for ann, m in masks:
            cid   = ann["category_id"]
            color = cat_colors.get(cid, (128,128,128))
            bgr   = (color[2], color[1], color[0])
            cname = cats[cid]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, cnts, -1, bgr, 2)

        stem = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), result)
        print(f"  saved → viz/{task_name}/{stem}.jpg")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TASK_TYPES = {
    "easy_semantic":   "semantic",
    "medium_instance": "instance",
    "hard_panoptic":   "instance",
    "cp1_holes":       "instance",
    "cp2_slice":       "instance",
    "cp5_occlusion":   "instance",
    "cp3_thin":        "semantic",
    "cp4_curb":        "semantic",
    "cp6_coverage":    "semantic",
}

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(TASK_TYPES)
    for task in targets:
        if task not in TASK_TYPES:
            print(f"[unknown] {task}"); continue
        print(f"\n=== {task} ===")
        if TASK_TYPES[task] == "semantic":
            viz_semantic(task)
        else:
            viz_instance(task)
    print(f"\nXong! Mở thư mục viz/ trong VSCode để xem ảnh.")

if __name__ == "__main__":
    main()
