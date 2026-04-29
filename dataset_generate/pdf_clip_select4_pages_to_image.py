"""
用 CLIP 给文档每页算视觉向量 → K-medoids(K=4) 选出 4 个聚类中心页 → 按页码排列后 2×2 拼接。
输出尺寸固定为 2×Letter 宽 × 2×Letter 高（与 Letter 同比例）。

PDF 目录（输入）:
  /data/docpc_project/datasets/pdfa/pages_4_30/{cat}
输出: .../pdfa/{category}/pos_target_for_deepseek_images_clip4
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ============ 全部参数（在此修改） ============
PAGES_BASE = Path("/data/docpc_project/datasets/pdfa_test/pages_4_30")
PROJECT_DIR = Path("/data/docpc_project/dataset_generate_new")
FINAL_SUBDIR = "pos_target_for_deepseek_images_clip4"
CATEGORIES = [
    "biology",
    "education",
    "finance",
    "government",
    "industrial",
    "legal",
    "research",
]
DPI = 150
CLIP_DPI = 72
# 使用本地模型：填写本地目录路径（需包含 config.json、preprocessor_config.json 等），不填则用下方 hub 名
LOCAL_CLIP_PATH = "/data/docpc_project/models/clip_vit_base_patch32"  # 例如: "/data/docpc_project/models/clip-vit-base-patch32"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 仅当 LOCAL_CLIP_PATH 为空时从 hub 下载
CLIP_BATCH_SIZE = 16
K_PAGES = 4
LETTER_PT = (612, 792)
WHITE_THRESHOLD = 250
CROP_MARGIN = 2
SUPPRESS_MUPDF_WARNINGS = True
# =============================================

LETTER_W_PX = int(LETTER_PT[0] * DPI / 72)
LETTER_H_PX = int(LETTER_PT[1] * DPI / 72)
LETTER_ASPECT = LETTER_PT[0] / LETTER_PT[1]


# --------------- 通用工具 ---------------

@contextmanager
def suppress_stderr():
    if not SUPPRESS_MUPDF_WARNINGS:
        yield
        return
    old = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        if sys.stderr != old:
            sys.stderr.close()
        sys.stderr = old


def page_to_pil(doc: fitz.Document, page_no: int, dpi: int = DPI) -> Image.Image | None:
    if page_no >= len(doc):
        return None
    page = doc[page_no]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_white_border(img: Image.Image, threshold: int = WHITE_THRESHOLD, margin: int = CROP_MARGIN) -> Image.Image:
    a = np.array(img)
    gray = np.max(a, axis=2) if a.ndim == 3 else a
    non_white = gray < threshold
    if not np.any(non_white):
        return img
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    rmax = min(a.shape[0], rmax + 1 + margin)
    cmin = max(0, cmin - margin)
    cmax = min(a.shape[1], cmax + 1 + margin)
    return img.crop((cmin, rmin, cmax, rmax))


def pad_to_aspect_ratio(
    img: Image.Image,
    target_ratio: float,
    fill: tuple = (255, 255, 255),
) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    current = w / h
    if abs(current - target_ratio) < 1e-6:
        return img
    if current > target_ratio:
        new_h = int(round(w / target_ratio))
        out = Image.new("RGB", (w, new_h), fill)
        y = (new_h - h) // 2
        out.paste(img, (0, y))
        return out
    else:
        new_w = int(round(h * target_ratio))
        out = Image.new("RGB", (new_w, h), fill)
        x = (new_w - w) // 2
        out.paste(img, (x, 0))
        return out


def process_one_page(doc: fitz.Document, page_no: int, dpi: int) -> Image.Image:
    blank = Image.new("RGB", (LETTER_W_PX, LETTER_H_PX), (255, 255, 255))
    if page_no >= len(doc):
        return blank
    pil = page_to_pil(doc, page_no, dpi)
    if pil is None:
        return blank
    pil = crop_white_border(pil)
    pil = pad_to_aspect_ratio(pil, LETTER_ASPECT)
    pil = pil.resize((LETTER_W_PX, LETTER_H_PX), Image.Resampling.LANCZOS)
    return pil


# --------------- CLIP + K-medoids 选页 ---------------

def load_clip(model_name: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = LOCAL_CLIP_PATH.strip() if LOCAL_CLIP_PATH else ""
    use_local = path != "" and os.path.isdir(path)
    if use_local:
        load_path = path
        kw = {"local_files_only": True}
        print(f"从本地加载 CLIP: {path}")
    else:
        load_path = model_name or CLIP_MODEL_NAME
        kw = {}
    model = CLIPModel.from_pretrained(load_path, **kw).to(device).eval()
    processor = CLIPProcessor.from_pretrained(load_path, **kw)
    print(f"CLIP 模型已加载: {load_path}  device={device}")
    return model, processor, device


@torch.no_grad()
def clip_page_embeddings(
    doc: fitz.Document,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> np.ndarray:
    """用低 DPI 渲染所有页面，返回 L2-归一化的 CLIP 视觉向量 (n_pages, dim)。"""
    images: list[Image.Image] = []
    for i in range(len(doc)):
        pil = page_to_pil(doc, i, dpi=CLIP_DPI)
        images.append(pil or Image.new("RGB", (224, 224), (255, 255, 255)))

    all_embs: list[np.ndarray] = []
    for start in range(0, len(images), CLIP_BATCH_SIZE):
        batch = images[start : start + CLIP_BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embs = model.get_image_features(**inputs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def kmedoids(dist: np.ndarray, k: int, max_iter: int = 100) -> list[int]:
    """在预计算的距离矩阵上跑 K-medoids，返回按升序排列的 medoid 下标。"""
    n = dist.shape[0]
    if n <= k:
        return list(range(n))

    # 贪心初始化：先选最中心点，再依次选离已选最远的点
    medoids = [int(np.argmin(dist.sum(axis=1)))]
    for _ in range(k - 1):
        min_d = dist[:, medoids].min(axis=1)
        medoids.append(int(np.argmax(min_d)))

    for _ in range(max_iter):
        labels = dist[:, medoids].argmin(axis=1)
        new_medoids: list[int] = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids.append(medoids[c])
            else:
                sub = dist[np.ix_(members, members)]
                new_medoids.append(int(members[sub.sum(axis=1).argmin()]))
        if sorted(new_medoids) == sorted(medoids):
            break
        medoids = new_medoids

    return sorted(medoids)


def select_pages(
    doc: fitz.Document,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    k: int = K_PAGES,
) -> list[int]:
    """CLIP + K-medoids 选出 k 个最具代表性的页码（升序）。"""
    n = len(doc)
    if n <= k:
        return list(range(n))
    embs = clip_page_embeddings(doc, model, processor, device)
    dist = 1.0 - embs @ embs.T
    np.fill_diagonal(dist, 0.0)
    return kmedoids(dist, k)


# --------------- 拼接 + 主流程 ---------------

def pdf_select4_to_image(
    pdf_path: str | Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    dpi: int = DPI,
) -> Image.Image:
    """
    CLIP + K-medoids 选出 4 个代表页 → 按页码排列 →
    裁白边 → 补白到 Letter 比例 → 缩放到 Letter 尺寸 → 2×2 拼接。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    with suppress_stderr():
        doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            raise ValueError("PDF 无有效页")
        selected = select_pages(doc, model, processor, device)
        imgs = [process_one_page(doc, i, dpi) for i in selected]
        blank = Image.new("RGB", (LETTER_W_PX, LETTER_H_PX), (255, 255, 255))
        while len(imgs) < 4:
            imgs.append(blank)
        total_w = 2 * LETTER_W_PX
        total_h = 2 * LETTER_H_PX
        out = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        out.paste(imgs[0], (0, 0))
        out.paste(imgs[1], (LETTER_W_PX, 0))
        out.paste(imgs[2], (0, LETTER_H_PX))
        out.paste(imgs[3], (LETTER_W_PX, LETTER_H_PX))
        return out
    finally:
        doc.close()


def run_all(categories: list[str]) -> None:
    model, processor, device = load_clip()
    for cat in categories:
        pdf_dir = PAGES_BASE / cat
        out_dir = PROJECT_DIR / "pdfa_test" / cat / FINAL_SUBDIR
        if not pdf_dir.is_dir():
            continue
        pdf_files = sorted([f for f in pdf_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
        if not pdf_files:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in tqdm(pdf_files, desc=f"[{cat}] CLIP→K-medoids→2x2"):
            out_path = out_dir / (pdf_path.stem + ".png")
            if out_path.exists():
                continue
            try:
                img = pdf_select4_to_image(pdf_path, model, processor, device, dpi=DPI)
                img.save(out_path, quality=95)
            except Exception as e:
                tqdm.write(f"跳过 {pdf_path}: {e}")


def main():
    print(f"选页：CLIP 视觉向量 → K-medoids(K={K_PAGES}) → 按页码排列")
    print(f"单页流程：裁白边 → 补白到 Letter 比例 → 缩放到 {LETTER_W_PX}x{LETTER_H_PX}")
    print(f"输出尺寸：{2 * LETTER_W_PX} x {2 * LETTER_H_PX}（Letter 四宫格）")
    run_all(CATEGORIES)
    print("完成")


if __name__ == "__main__":
    main()
