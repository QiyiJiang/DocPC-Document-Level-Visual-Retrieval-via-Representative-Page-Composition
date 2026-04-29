"""
读取 7 个类别的 PDF，将所有页转为图片并生成 metadata JSON。

输入: /data/docpc_project/datasets/pdfa_test/pages_4_30/{cat}
输出图片: /data/docpc_project/datasets/pdfa_test/image_page_{cat}/
输出JSON: /data/docpc_project/datasets/pdfa_test/image_page_metadata_{cat}.json
"""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

# ============ 参数 ============
PAGES_BASE = Path("/data/docpc_project/datasets/pdfa_test/pages_4_30")
OUT_BASE = Path("/data/docpc_project/datasets/pdfa_test")
CATEGORIES = [
    "biology",
    "education",
    "finance",
    "government",
    "industrial",
    "legal",
    "research",
]
MAX_PAGES = None  # None = 不限制，处理所有页
DPI = 150
LETTER_PT = (612, 792)
WHITE_THRESHOLD = 250
CROP_MARGIN = 2
SUPPRESS_MUPDF_WARNINGS = True
# =============================

LETTER_W_PX = int(LETTER_PT[0] * DPI / 72)
LETTER_H_PX = int(LETTER_PT[1] * DPI / 72)
LETTER_ASPECT = LETTER_PT[0] / LETTER_PT[1]


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


def crop_white_border(
    img: Image.Image, threshold: int = WHITE_THRESHOLD, margin: int = CROP_MARGIN
) -> Image.Image:
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


def process_one_page(
    doc: fitz.Document,
    page_no: int,
    dpi: int,
) -> Image.Image:
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


def process_pdf_and_collect_metadata(
    pdf_path: Path,
    out_dir: Path,
    records: list,
    max_pages: int = MAX_PAGES,
    skip_existing: bool = True,
) -> None:
    stem = pdf_path.stem
    with suppress_stderr():
        doc = fitz.open(pdf_path)
    try:
        n_pages = min(len(doc), max_pages) if max_pages else len(doc)
        if n_pages == 0:
            return
        for page_no in range(n_pages):
            img = process_one_page(doc, page_no, DPI)
            page_fname = f"{stem}_{page_no}.png"
            out_path = out_dir / page_fname
            if not skip_existing or not out_path.exists():
                img.save(out_path, quality=95)
            records.append({
                "page_name": page_fname,
                "document_name": f"{stem}.png",
            })
        records.append({
            "page_name": f"{stem}.png",
            "document_name": f"{stem}.png",
        })
    finally:
        doc.close()


def main():
    for cat in CATEGORIES:
        pdf_dir = PAGES_BASE / cat
        if not pdf_dir.is_dir():
            print(f"[{cat}] PDF 目录不存在: {pdf_dir}，跳过")
            continue

        out_dir = OUT_BASE / f"image_page_{cat}"
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        pdf_files = sorted(
            [f for f in pdf_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
        )
        pages_desc = "所有页" if MAX_PAGES is None else f"前 {MAX_PAGES} 页"
        print(f"[{cat}] 共 {len(pdf_files)} 个 PDF，每个取{pages_desc}")

        for pdf_path in tqdm(pdf_files, desc=f"[{cat}] PDF→页图"):
            try:
                process_pdf_and_collect_metadata(pdf_path, out_dir, records)
            except Exception as e:
                tqdm.write(f"跳过 {pdf_path}: {e}")

        out_json = OUT_BASE / f"image_page_metadata_{cat}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[{cat}] 共 {len(records)} 条记录，已写入 {out_json}")

    print("全部完成")


if __name__ == "__main__":
    main()
