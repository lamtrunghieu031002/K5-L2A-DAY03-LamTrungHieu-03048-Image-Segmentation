#!/usr/bin/env python3
"""
Gợi ý mask tự động cho Day 3 bằng Mask2Former Swin-L (CPU được).

  Ảnh đường phố (BDD, nhãn kiểu Cityscapes)  -> mask2former-swin-large-cityscapes-semantic
      easy_semantic, cp3_thin, cp4_curb, cp6_coverage  -> Segmentation mask 1.1
  Ảnh COCO (nhãn kiểu COCO / COCO-panoptic)  -> mask2former-swin-large-coco-panoptic
      medium_instance, cp1_holes, cp2_slice, cp5_occlusion -> COCO 1.0 (instance)
      hard_panoptic (stuff gộp theo class + từng thing)     -> COCO 1.0

Quy tắc áp dụng:
  * Chỉ giữ class có trong classes.json của task; pixel khác để trống (background).
  * Instance: mỗi vật một mask, không chồng lấn (chỉ phần nhìn thấy); một vật bị che
    có thể gồm nhiều mảng rời nhưng vẫn là MỘT annotation.
  * Lỗ bên trong vật (kính, khe) được lấp vào mask vật (quy tắc cp1_holes).
  * Mask lưu dạng RLE (như CVAT) để giữ đúng lỗ của stuff (vd. mảng trời giữa tán cây).

Dùng:  python3 generate_annotations_m2f.py [task ...]
"""
import io, json, sys, zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "submissions"
torch.set_grad_enabled(False)

CITY = "facebook/mask2former-swin-large-cityscapes-semantic"
COCO = "facebook/mask2former-swin-large-coco-panoptic"
SIZES = {CITY: {"shortest_edge": 1024, "longest_edge": 2048},
         COCO: {"shortest_edge": 800, "longest_edge": 1333}}

# tên class COCO-panoptic -> tên class của bài (khớp catid2name của hard_panoptic)
COCO2TASK = {"person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle",
             "bus": "bus", "truck": "truck", "traffic light": "traffic light",
             "road": "road", "pavement-merged": "sidewalk", "building-other-merged": "building",
             "tree-merged": "vegetation", "sky-other-merged": "sky"}

# Cityscapes: cỏ/đất/núi trọc là `terrain` (không thuộc task), `vegetation` chỉ là cây/bụi mọc đứng
GROUND_COVER = {"mountain-merged", "grass-merged", "dirt-merged", "sand", "gravel",
                "rock-merged", "playingfield"}

INSTANCE_SCORE = 0.5   # ngưỡng tin cậy cho object
MIN_AREA = 40          # bỏ mảnh nhỏ hơn (pixel)

_models = {}


def load(name):
    if name not in _models:
        _models.clear()                              # giữ 1 model trong RAM
        proc = Mask2FormerImageProcessor.from_pretrained(name)
        proc.size = SIZES[name]
        model = Mask2FormerForUniversalSegmentation.from_pretrained(name).eval()
        _models[name] = (proc, model)
        print(f"[load] {name}")
    return _models[name]


def run(name, pil):
    proc, model = load(name)
    return proc, model, model(**proc(images=pil, return_tensors="pt"))


# ---------------------------------------------------------------------------
# semantic: xác suất class theo pixel, trung bình ảnh gốc + ảnh lật ngang
# ---------------------------------------------------------------------------
def semantic_probs(pil):
    W, H = pil.size
    acc = None
    for flip in (False, True):
        img = pil.transpose(Image.FLIP_LEFT_RIGHT) if flip else pil
        _, _, out = run(CITY, img)
        cls = out.class_queries_logits.softmax(-1)[..., :-1]          # B,Q,C
        msk = out.masks_queries_logits.sigmoid()                      # B,Q,h,w
        msk = torch.nn.functional.interpolate(msk, size=(H, W), mode="bilinear", align_corners=False)
        p = torch.einsum("bqc,bqhw->bchw", cls, msk)[0]
        if flip:
            p = p.flip(-1)
        acc = p if acc is None else acc + p
    return acc / 2                                                   # C,H,W


def coco_exclusions(pil):
    """Dùng model COCO để tìm (a) mui xe của chính mình (ego vehicle, Cityscapes/BDD để
    'ignore') và (b) vùng bus/truck — để không gán nhầm thành road/car."""
    W, H = pil.size
    proc, model, out = run(COCO, pil)
    res = proc.post_process_panoptic_segmentation(out, threshold=0.5, target_sizes=[(H, W)])[0]
    seg = res["segmentation"].numpy()
    hood = np.zeros((H, W), bool); big_vehicle = np.zeros((H, W), bool); ground = np.zeros((H, W), bool)
    for s in res["segments_info"]:
        name = model.config.id2label[s["label_id"]]
        m = seg == s["id"]
        if name in ("car", "truck") and m[-1].mean() > 0.3 and m[: H // 2].sum() == 0:
            hood |= m                       # chạm đáy ảnh, phủ >30% bề ngang, không lên nửa trên
        elif name in ("bus", "truck", "train"):
            big_vehicle |= m
        elif name in GROUND_COVER:
            ground |= m
    return hood, big_vehicle, ground


def tree_clusters(pil, region):
    """Trong vùng đồi cỏ/đất, tách các bụi/cụm cây: tán cây sẫm ánh xanh lạnh
    (B − R ≥ 14, LAB b* ≤ 122), còn cỏ khô/đất ngả nâu (R ≥ B)."""
    import cv2
    img = np.array(pil)
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.5), cv2.COLOR_RGB2LAB).astype(np.float32)
    m = region & ((f[..., 2] - f[..., 0]) >= 14) & (lab[..., 2] <= 122)
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=1), iterations=2)
    m = ndimage.binary_fill_holes(m) & region
    lab_id, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab_id, range(1, n + 1))
    return np.isin(lab_id, np.where(sizes >= 80)[0] + 1)


def semantic_labels(pil, classes, excl=None):
    """-> HxW mảng chỉ số class của task (-1 = để trống)."""
    _, model = load(CITY)
    id2label = model.config.id2label
    probs = semantic_probs(pil)
    pred = probs.argmax(0).numpy()
    out = np.full(pred.shape, -1, np.int32)
    for cid, name in id2label.items():
        if name in classes:
            out[pred == int(cid)] = classes.index(name)
    if excl is not None:
        hood, big_vehicle, ground = excl
        out[hood] = -1                                   # ego vehicle: không thuộc class nào
        if "vegetation" in classes:                      # đồi cỏ/đất = terrain, không phải vegetation
            veg = classes.index("vegetation")
            hill = ground & (out == veg)
            out[hill & ~tree_clusters(pil, hill)] = -1   # chỉ giữ bụi/cụm cây trên đồi
        if "car" in classes and "bus" not in classes:    # xe buýt/tải không phải car
            car = classes.index("car")
            out[big_vehicle & (out == car)] = -1
    return out


# ---------------------------------------------------------------------------
# COCO model: instance / panoptic
# ---------------------------------------------------------------------------
def fill_holes(m):
    return ndimage.binary_fill_holes(m)


def coco_segments(pil, keep, panoptic_stuff=()):
    """-> list[(task_class, mask, score)] không chồng lấn."""
    W, H = pil.size
    proc, model, out = run(COCO, pil)
    id2label = model.config.id2label
    if panoptic_stuff:
        fuse = {int(i) for i, n in id2label.items() if COCO2TASK.get(n) in panoptic_stuff}
        res = proc.post_process_panoptic_segmentation(
            out, threshold=INSTANCE_SCORE, label_ids_to_fuse=fuse, target_sizes=[(H, W)])[0]
    else:
        res = proc.post_process_instance_segmentation(
            out, threshold=INSTANCE_SCORE, target_sizes=[(H, W)])[0]
    seg = res["segmentation"].numpy()
    items = []
    for s in res["segments_info"]:
        name = COCO2TASK.get(id2label[s["label_id"]])
        if name not in keep:
            continue
        m = seg == s["id"]
        if m.sum() >= MIN_AREA:
            items.append([name, m, float(s["score"])])
    # lấp lỗ cho thing (kính/khe nằm trong vật), không lấn sang vật khác đã có
    taken = np.zeros((H, W), bool)
    for it in items:
        if it[0] not in panoptic_stuff:
            taken |= it[1]
    for it in items:
        if it[0] not in panoptic_stuff:
            filled = fill_holes(it[1])
            it[1] = it[1] | (filled & ~taken)
    return items


# ---------------------------------------------------------------------------
# định dạng xuất
# ---------------------------------------------------------------------------
def rle_fast(mask):
    """COCO uncompressed RLE (column-major) giống CVAT export."""
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def write_zip(task, files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items():
            zf.writestr(k, v)
    (OUT / f"{task}.zip").write_bytes(buf.getvalue())


def build_semantic(task, base):
    cfg = json.loads((base / "classes.json").read_text())
    classes, colors = cfg["classes"], cfg["colors"]
    files = {"labelmap.txt": "# label:color_rgb:parts:actions\nbackground:0,0,0::\n" +
             "".join(f"{c}:{','.join(map(str, colors[c]))}::\n" for c in classes)}
    names = []
    imgs = sorted((base / "images").glob("*.jpg"))
    excl = {p: coco_exclusions(Image.open(p).convert("RGB")) for p in imgs}   # pass 1: COCO
    for p, (hood, bv, gr) in excl.items():
        print(f"  {p.name}: ego-hood {hood.mean():.1%}, bus/truck {bv.mean():.1%}, ground-cover {gr.mean():.1%}")
    for img in imgs:
        pil = Image.open(img).convert("RGB")
        lab = semantic_labels(pil, classes, excl[img])
        rgb = np.zeros((*lab.shape, 3), np.uint8)
        for i, c in enumerate(classes):
            rgb[lab == i] = colors[c]
        b = io.BytesIO(); Image.fromarray(rgb).save(b, "PNG")
        files[f"SegmentationClass/{img.stem}.png"] = b.getvalue()
        names.append(img.stem)
        share = {c: round(float((lab == i).mean()), 3) for i, c in enumerate(classes)}
        print(f"  {img.name}: {share}")
    files["ImageSets/Segmentation/default.txt"] = "\n".join(names) + "\n"
    write_zip(task, files)


def build_coco(task, base, panoptic):
    cfg = json.loads((base / "classes.json").read_text())
    classes = cfg["classes"]
    stuff = set(cfg.get("stuff", [])) if panoptic else set()
    cats = [{"id": i + 1, "name": c, "supercategory": ""} for i, c in enumerate(classes)]
    cid = {c: i + 1 for i, c in enumerate(classes)}
    images, anns = [], []
    for img in sorted((base / "images").glob("*.jpg")):
        pil = Image.open(img).convert("RGB")
        W, H = pil.size
        iid = len(images) + 1
        images.append({"id": iid, "file_name": img.name, "width": W, "height": H})
        items = coco_segments(pil, set(classes), panoptic_stuff=stuff)
        if panoptic:                                  # stuff: một annotation / class
            merged = {}
            for name, m, sc in items:
                if name in stuff:
                    merged[name] = merged.get(name, np.zeros_like(m)) | m
            items = [[n, m, 1.0] for n, m in merged.items()] + [it for it in items if it[0] not in stuff]
        counts = {}
        for name, m, sc in items:
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cid[name],
                         "segmentation": rle_fast(m), "area": int(m.sum()), "bbox": bbox(m),
                         "iscrowd": 1, "attributes": {"occluded": False}})
            counts[name] = counts.get(name, 0) + 1
        print(f"  {img.name}: {counts}")
    coco = {"licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {"description": "", "version": "", "year": ""},
            "categories": cats, "images": images, "annotations": anns}
    write_zip(task, {"annotations/instances_default.json": json.dumps(coco)})


TASKS = {
    "easy_semantic": ("tiers/easy_semantic", "semantic"),
    "cp3_thin": ("checkpoints/cp3_thin", "semantic"),
    "cp4_curb": ("checkpoints/cp4_curb", "semantic"),
    "cp6_coverage": ("checkpoints/cp6_coverage", "semantic"),
    "medium_instance": ("tiers/medium_instance", "instance"),
    "cp1_holes": ("checkpoints/cp1_holes", "instance"),
    "cp2_slice": ("checkpoints/cp2_slice", "instance"),
    "cp5_occlusion": ("checkpoints/cp5_occlusion", "instance"),
    "hard_panoptic": ("tiers/hard_panoptic", "panoptic"),
}

if __name__ == "__main__":
    for task in (sys.argv[1:] or TASKS):
        rel, kind = TASKS[task]
        print(f"== {task} ({kind})")
        if kind == "semantic":
            build_semantic(task, DATA / rel)
        else:
            build_coco(task, DATA / rel, panoptic=(kind == "panoptic"))
        print(f"[OK] submissions/{task}.zip")
