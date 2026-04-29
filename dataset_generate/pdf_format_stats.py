"""
递归扫描指定目录下所有 PDF，统计页面尺寸（版式）分布。
以每份 PDF 的第一页为准；尺寸按「短边 x 长边」归一化后与常见版式匹配。
"""
from pathlib import Path

import fitz  # PyMuPDF

# 常见版式（短边, 长边）单位：点 pt，约 1pt ≈ 0.35mm，允许 ±2pt 误差
FORMATS = [
    ("A4", (595, 842)),
    ("Letter", (612, 792)),
    ("Legal", (612, 1008)),
    ("A3", (842, 1191)),
    ("A5", (420, 595)),
    ("Tabloid", (792, 1224)),
    ("Executive", (522, 756)),
]
TOLERANCE = 2


def normalize_size(w: float, h: float) -> tuple[int, int]:
    """归一化为 (短边, 长边) 的整数点。"""
    a, b = round(w), round(h)
    return (min(a, b), max(a, b)) if a != b else (a, b)


def match_format(w: float, h: float) -> str:
    """匹配已知版式，否则返回 'Other (WxH)'。"""
    short, long_ = normalize_size(w, h)
    for name, (s, l_) in FORMATS:
        if abs(short - s) <= TOLERANCE and abs(long_ - l_) <= TOLERANCE:
            return name
    return f"Other ({short}x{long_})"


def main():
    root = Path("/data/docpc_project/datasets/pdfa/pages_4_30")
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return

    pdf_files = sorted(root.rglob("*.pdf"))
    if not pdf_files:
        print(f"未找到 PDF: {root}")
        return

    counts: dict[str, int] = {}
    for path in pdf_files:
        try:
            doc = fitz.open(path)
            if len(doc) == 0:
                doc.close()
                continue
            page = doc[0]
            rect = page.rect  # MediaBox 尺寸，单位 pt
            w, h = rect.width, rect.height
            doc.close()
        except Exception:
            continue
        fmt = match_format(w, h)
        counts[fmt] = counts.get(fmt, 0) + 1

    total = sum(counts.values())
    print(f"目录: {root}")
    print(f"扫描 PDF 数: {len(pdf_files)}, 成功读取首页: {total}")
    print("\n版式统计（按数量从高到低）:")
    for fmt, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total if total else 0
        print(f"  {fmt}: {n} ({pct:.1f}%)")
    if counts:
        most = max(counts.items(), key=lambda x: x[1])
        print(f"\n使用最多的版式: {most[0]} ({most[1]} 份)")


if __name__ == "__main__":
    main()
