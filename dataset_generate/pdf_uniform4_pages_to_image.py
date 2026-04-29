"""
将 PDF「均匀 4 页」按「每页裁白边 → 补白到 Letter 比例 → 缩放到 Letter 尺寸」后 2x2 拼接。
例如 10 页取第 1、4、7、10 页（0-based 为 0、3、6、9）。
输出尺寸固定为 2×Letter 宽 × 2×Letter 高（与 Letter 同比例）。

PDF 目录（输入）:
  /data/docpc_project/datasets/pdfa_test/pages_4_30/{cat}_pos_target_for_deepseek
输出: .../pdfa_test/{category}/pos_target_for_deepseek_images_uniform4
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
FINAL_SUBDIR = "pos_target_for_deepseek_images_uniform4"
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
# Letter 尺寸（PDF 点），比例 612/792
LETTER_PT = (612, 792)  # (宽, 高) 竖版
# 裁剪白边
WHITE_THRESHOLD = 250
CROP_MARGIN = 2
SUPPRESS_MUPDF_WARNINGS = True
# =============================================

# 当前 DPI 下单页 Letter 像素尺寸
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
    """将 PDF 第 page_no 页（0-based）渲染为 PIL Image。"""
    if page_no >= len(doc):
        return None
    page = doc[page_no]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_white_border(img: Image.Image, threshold: int = WHITE_THRESHOLD, margin: int = CROP_MARGIN) -> Image.Image:
    """裁剪图像四周白边。"""
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
    """
    将图片补白到目标宽高比：过宽则上下补白，过高则左右补白，居中粘贴。
    target_ratio = 宽/高（如 Letter 612/792）。
    """
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
    """
    单页流程：渲染 → 裁白边 → 补白到 Letter 比例 → 缩放到 Letter 像素尺寸。
    若页不存在或渲染失败返回一张空白 Letter 图。
    """
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


def uniform4_indices(n: int) -> list[int]:
    """
    总页数为 n 时，均匀取 4 个页码（0-based）。
    例如 n=10 → [0, 3, 6, 9]（即第 1、4、7、10 页）。
    """
    if n <= 0:
        return [0, 0, 0, 0]
    if n == 1:
        return [0, 0, 0, 0]
    # 均匀 4 点：i 取 0, 1, 2, 3 对应 round(i * (n-1) / 3)
    return [min(round(i * (n - 1) / 3), n - 1) for i in range(4)]


def pdf_uniform4_to_image(
    pdf_path: str | Path,
    dpi: int = DPI,
) -> Image.Image:
    """
    将 PDF 均匀 4 页（第 1、1/3、2/3、最后 1 页）按「裁白边 → 补白到 Letter 比例 → 缩放到 Letter 尺寸」后 2x2 拼接。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    with suppress_stderr():
        doc = fitz.open(pdf_path)
    try:
        n = len(doc)
        if n == 0:
            raise ValueError("PDF 无有效页")

        indices = uniform4_indices(n)
        imgs = [process_one_page(doc, i, dpi) for i in indices]
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
    """
    按类别遍历 PDF：每份 PDF 均匀 4 页按「裁白边 → 补白到 Letter 比例 → 缩放到 Letter」后 2x2 拼接，
    保存到 pos_target_for_deepseek_images_uniform4；已存在的 PNG 跳过。
    """
    for cat in categories:
        pdf_dir = PAGES_BASE / f"{cat}"
        out_dir = PROJECT_DIR / "pdfa_test" / cat / FINAL_SUBDIR
        if not pdf_dir.is_dir():
            continue
        pdf_files = sorted([f for f in pdf_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
        if not pdf_files:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in tqdm(pdf_files, desc=f"[{cat}] uniform4 2x2"):
            out_path = out_dir / (pdf_path.stem + ".png")
            if out_path.exists():
                continue
            try:
                img = pdf_uniform4_to_image(pdf_path, dpi=DPI)
                img.save(out_path, quality=95)
            except Exception as e:
                tqdm.write(f"跳过 {pdf_path}: {e}")


def main():
    print("单页流程：裁白边 → 补白到 Letter 比例 → 缩放到 Letter → 均匀 4 页 2x2 拼接")
    print(f"均匀取页：第 1 页、约 1/3 处、约 2/3 处、最后 1 页（如 10 页即第 1、4、7、10 页）")
    print(f"输出尺寸：{2 * LETTER_W_PX} x {2 * LETTER_H_PX}（Letter 四宫格）")
    print(f"输出目录：.../pdfa_test/{{category}}/{FINAL_SUBDIR}")
    run_all(CATEGORIES)
    print("完成")


if __name__ == "__main__":
    main()
