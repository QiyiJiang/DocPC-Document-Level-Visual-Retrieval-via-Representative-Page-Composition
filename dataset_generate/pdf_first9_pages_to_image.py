"""
将 PDF 前 9 页（不足则循环使用已有页）按「裁白边 → 补白到 Letter 比例 → 缩放到 Letter」后 3×3 拼接。
输出尺寸 = 3×Letter 宽 × 3×Letter 高。

输入: PAGES_BASE / {cat}
输出: PROJECT_DIR / pdfa / {cat} / pos_target_for_deepseek_images_first9
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

# ============ 全部参数（在此修改） ============
PAGES_BASE = Path("/data/docpc_project/datasets/pdfa_test/pages_4_30")
PROJECT_DIR = Path("/data/docpc_project/dataset_generate_new")
FINAL_SUBDIR = "pos_target_for_deepseek_images_first9"
CATEGORIES = [
    "biology", "education", "finance", "government",
    "industrial", "legal", "research",
]
DPI = 150
LETTER_PT = (612, 792)
WHITE_THRESHOLD = 250
CROP_MARGIN = 2
SUPPRESS_MUPDF_WARNINGS = True
# =============================================

LETTER_W_PX = int(LETTER_PT[0] * DPI / 72)
LETTER_H_PX = int(LETTER_PT[1] * DPI / 72)
LETTER_ASPECT = LETTER_PT[0] / LETTER_PT[1]
N_NEED = 9
GRID = (3, 3)


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


def pad_to_aspect_ratio(img: Image.Image, target_ratio: float, fill: tuple = (255, 255, 255)) -> Image.Image:
    w, h = img.size
    if h <= 0:
        return img
    current = w / h
    if abs(current - target_ratio) < 1e-6:
        return img
    if current > target_ratio:
        new_h = int(round(w / target_ratio))
        out = Image.new("RGB", (w, new_h), fill)
        out.paste(img, (0, (new_h - h) // 2))
        return out
    else:
        new_w = int(round(h * target_ratio))
        out = Image.new("RGB", (new_w, h), fill)
        out.paste(img, ((new_w - w) // 2, 0))
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


def pdf_first9_to_image(pdf_path: str | Path, dpi: int = DPI) -> Image.Image:
    """前 9 页，不足则循环使用前面页，3×3 拼接。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    with suppress_stderr():
        doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            raise ValueError("PDF 无有效页")
        n = len(doc)
        indices = [i % n for i in range(N_NEED)]
        imgs = [process_one_page(doc, i, dpi) for i in indices]
        total_w = GRID[0] * LETTER_W_PX
        total_h = GRID[1] * LETTER_H_PX
        out = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        for idx, img in enumerate(imgs):
            row, col = idx // GRID[1], idx % GRID[1]
            out.paste(img, (col * LETTER_W_PX, row * LETTER_H_PX))
        return out
    finally:
        doc.close()


def run_all(categories: list[str]) -> None:
    for cat in categories:
        pdf_dir = PAGES_BASE / f"{cat}"
        out_dir = PROJECT_DIR / "pdfa_test" / cat / FINAL_SUBDIR
        if not pdf_dir.is_dir():
            continue
        pdf_files = sorted([f for f in pdf_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
        if not pdf_files:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in tqdm(pdf_files, desc=f"[{cat}] first9 3x3"):
            out_path = out_dir / (pdf_path.stem + ".png")
            if out_path.exists():
                continue
            try:
                img = pdf_first9_to_image(pdf_path, dpi=DPI)
                img.save(out_path, quality=95)
            except Exception as e:
                tqdm.write(f"跳过 {pdf_path}: {e}")


def main():
    print(f"first9: 9 页 3×3 拼接，不足 9 页循环使用前若干页")
    print(f"输出尺寸：{3 * LETTER_W_PX} x {3 * LETTER_H_PX}")
    run_all(CATEGORIES)
    print("完成")


if __name__ == "__main__":
    main()
