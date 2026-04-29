"""
从 PDF 提取文本并保存为 .txt。
支持两种方式：
1. MinerU（推荐）：布局分析 + 公式/表格，输出质量更好，适合论文/复杂版式。
2. PyMuPDF（fallback）：简单快速，适合仅需纯文本或 MinerU 不可用时。
多卡时可将 PDF 分片，按 GPU_IDS 绑卡；NUM_WORKERS 可大于 GPU 数（单卡多进程），显存不足可减小。
"""
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ============ 配置（在此修改） ============
PDF_DIR = "/data/docpc_project/datasets/pdfa_test/pages_4_30"
OUT_DIR = "/data/docpc_project/datasets/pdfa_test_text_pymupdf"

# 提取方式: "mineru" | "pymupdf" | "mineru_then_pymupdf"（MinerU 失败时用 PyMuPDF 兜底）
METHOD = "pymupdf"

# MinerU 批量模式：True=整目录一次调用 mineru -p PDF_DIR -o TEMP（推荐，可看进度）；False=逐篇调用（慢且易显卡住）
MINERU_BATCH = True

# 最多处理的 PDF 数量，None 表示不限制（先试跑可设为 1000）
MAX_PDF_COUNT = 10000

# MinerU 临时目录，处理完可删
MINERU_TEMP_BASE = "/data/docpc_project/datasets/arxiv_mineru_temp"

# MinerU 模型来源，无法访问 HF 时设为 "modelscope"；可选 "huggingface" | "modelscope" | "local"
MINERU_MODEL_SOURCE = "modelscope"

# 使用的 GPU 编号（仅 MinerU 时绑卡；PyMuPDF 用 NUM_WORKERS 做多进程并行）
GPU_IDS = [1, 2, 3, 4, 5, 6]

# 并发 worker 数。PyMuPDF 时即并行处理的进程数；MinerU 时可大于 GPU 数
NUM_WORKERS = 6
# =============================================


def _mineru_env(cuda_device: int | None = None):
    """子进程环境变量，注入 MINERU_MODEL_SOURCE；cuda_device 为物理 GPU 编号时设置 CUDA_VISIBLE_DEVICES。"""
    env = {**os.environ, "MINERU_MODEL_SOURCE": MINERU_MODEL_SOURCE}
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    return env


def extract_pymupdf(pdf_path: str, txt_path: str) -> bool:
    """用 PyMuPDF 提取纯文本。"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        return True
    except Exception as e:
        print(f"  PyMuPDF 失败 {pdf_path}: {e}")
        return False


def extract_mineru(pdf_path: str, txt_path: str, temp_base: str) -> bool:
    """
    用 MinerU 命令行提取，得到 .md 后转存为 .txt（或保留 .md 内容）。
    单文件调用: mineru -p <pdf> -o <out_dir>，输出为 <out_dir>/<pdf_basename>.md
    """
    pdf_path = Path(pdf_path).resolve()
    base = pdf_path.stem  # 不含扩展名
    out_sub = Path(temp_base) / base
    out_sub.mkdir(parents=True, exist_ok=True)

    try:
        # 不 capture_output，便于在终端看到 MinerU 进度（否则会显得卡住）
        subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(out_sub)],
            check=True,
            timeout=600,
            env=_mineru_env(),
        )
    except subprocess.CalledProcessError as e:
        print(f"  MinerU 执行失败 {pdf_path}: {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  MinerU 超时 {pdf_path}")
        return False
    except FileNotFoundError:
        print("  MinerU 未安装或不在 PATH，请: pip install mineru 并配置")
        return False

    # MinerU 实际写入: output_dir / pdf_file_name / parse_method / {basename}.md
    # 即 out_sub / base / "auto"|"vlm"|"hybrid_auto" 等 / base.md
    md_file = None
    for p in out_sub.rglob("*.md"):
        if p.stem == base or p.name == f"{base}.md":
            md_file = p
            break
    if not md_file or not md_file.exists():
        print(f"  未找到 MinerU 输出 .md: {out_sub}")
        return False

    try:
        with open(md_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"  读取 MinerU 输出失败: {e}")
        return False


def _run_mineru_one_chunk(
    worker_id: int,
    gpu_id: int,
    pdf_dir: Path,
    chunk: list[str],
    temp_base: Path,
) -> None:
    """单 worker：为 chunk 建子目录、跑 mineru，绑定物理 GPU gpu_id。"""
    work_out = temp_base / f"worker_{worker_id}"
    work_pdf = temp_base / f"worker_{worker_id}_pdf_subset"
    work_out.mkdir(parents=True, exist_ok=True)
    if work_pdf.exists():
        shutil.rmtree(work_pdf)
    work_pdf.mkdir(parents=True)
    for f in chunk:
        (work_pdf / f).symlink_to(pdf_dir / f)
    subprocess.run(
        ["mineru", "-p", str(work_pdf), "-o", str(work_out)],
        check=True,
        timeout=86400,
        env=_mineru_env(cuda_device=gpu_id),
    )


def _collect_md_to_txt(temp_base: Path, out_dir: Path) -> int:
    """从 temp_base 下所有 worker 目录收集 .md，复制为 out_dir/*.txt。"""
    md_list = list(temp_base.rglob("*.md"))
    copied = 0
    for md_file in tqdm(md_list, desc="收集 .md → .txt"):
        stem = md_file.stem
        txt_path = out_dir / f"{stem}.txt"
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            txt_path.write_text(content, encoding="utf-8")
            copied += 1
        except Exception as e:
            print(f"  跳过 {md_file}: {e}")
    return copied


def extract_mineru_batch(
    pdf_dir: str,
    pdf_files: list[str],
    out_dir: str,
    temp_base: str,
) -> None:
    """
    批量：按 NUM_WORKERS 分片，每片在一个 mineru 进程中处理（绑定不同 GPU），最后合并 .md → .txt。
    """
    pdf_dir = Path(pdf_dir).resolve()
    temp_base = Path(temp_base)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_base.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, min(NUM_WORKERS, len(pdf_files)))
    # 均分 PDF 到各 worker
    chunk_size = (len(pdf_files) + n_workers - 1) // n_workers
    chunks = [
        pdf_files[i : i + chunk_size]
        for i in range(0, len(pdf_files), chunk_size)
    ]
    if len(chunks) > n_workers:
        chunks = chunks[:n_workers]

    if n_workers == 1:
        work_pdf = temp_base / (temp_base.name + "_pdf_subset")
        if work_pdf.exists():
            shutil.rmtree(work_pdf)
        work_pdf.mkdir(parents=True)
        for f in pdf_files:
            (work_pdf / f).symlink_to(pdf_dir / f)
        gpu_id = GPU_IDS[0] if GPU_IDS else None
        print(f"MinerU 批量: -p {work_pdf} -o {temp_base}（共 {len(pdf_files)} 个 PDF，GPU {gpu_id}）")
        print("进度见下方 MinerU 输出…")
        try:
            subprocess.run(
                ["mineru", "-p", str(work_pdf), "-o", str(temp_base)],
                check=True,
                timeout=86400,
                env=_mineru_env(cuda_device=gpu_id),
            )
        except subprocess.CalledProcessError as e:
            print(f"MinerU 执行失败: {e}")
            return
        except subprocess.TimeoutExpired:
            print("MinerU 超时（默认 24h）")
            return
        except FileNotFoundError:
            print("MinerU 未安装或不在 PATH，请: pip install mineru 并配置")
            return
        copied = _collect_md_to_txt(temp_base, out_dir)
        print(f"已复制 {copied} 个 .md → {out_dir}")
        return

    # 多 worker 并发，worker 按 GPU_IDS 轮询绑卡（可单卡多进程）
    n_gpus = len(GPU_IDS)
    print(f"MinerU 多卡并发: {n_workers} 个 worker，GPU {GPU_IDS}，共 {len(pdf_files)} 个 PDF")
    for i, c in enumerate(chunks):
        gpu_id = GPU_IDS[i % n_gpus]
        print(f"  worker_{i}（物理 GPU {gpu_id}）: {len(c)} 个 PDF")
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    _run_mineru_one_chunk,
                    i,
                    GPU_IDS[i % n_gpus],
                    pdf_dir,
                    chunks[i],
                    temp_base,
                ): i
                for i in range(len(chunks))
            }
            with tqdm(total=n_workers, desc="MinerU workers", unit="worker") as pbar:
                for fut in as_completed(futures):
                    wid = futures[fut]
                    try:
                        fut.result()
                        pbar.set_postfix_str(f"worker_{wid} 完成")
                    except subprocess.CalledProcessError as e:
                        print(f"\nWorker {wid} MinerU 失败: {e}")
                    except subprocess.TimeoutExpired:
                        print(f"\nWorker {wid} MinerU 超时")
                    except FileNotFoundError:
                        print("\nMinerU 未安装或不在 PATH")
                        return
                    pbar.update(1)
    except Exception as e:
        print(f"并发执行异常: {e}")
        return

    copied = _collect_md_to_txt(temp_base, out_dir)
    print(f"已复制 {copied} 个 .md → {out_dir}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f"未找到 PDF: {PDF_DIR}")
        return
    if MAX_PDF_COUNT is not None:
        pdf_files = pdf_files[:MAX_PDF_COUNT]
        print(f"限制处理数量: {len(pdf_files)} 个 PDF（MAX_PDF_COUNT={MAX_PDF_COUNT}）")
    else:
        print(f"待处理 PDF: {len(pdf_files)} 个")

    use_mineru = METHOD in ("mineru", "mineru_then_pymupdf")

    if use_mineru and MINERU_BATCH:
        # 批量：一次 mineru 处理（仅当前 pdf_files 列表），再收集 .md → .txt
        extract_mineru_batch(PDF_DIR, pdf_files, OUT_DIR, MINERU_TEMP_BASE)
        if os.path.isdir(MINERU_TEMP_BASE):
            print(f"MinerU 临时目录可手动删除: {MINERU_TEMP_BASE}")
        return

    if use_mineru:
        os.makedirs(MINERU_TEMP_BASE, exist_ok=True)

    n_workers = max(1, min(NUM_WORKERS, len(pdf_files)))
    if METHOD == "pymupdf" and n_workers > 1:
        # PyMuPDF 多 worker 并行（不占 GPU，多进程加速）
        def _pymupdf_one(pdf_file: str) -> bool:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            txt_path = os.path.join(OUT_DIR, pdf_file.replace(".pdf", ".txt"))
            return extract_pymupdf(pdf_path, txt_path)

        print(f"PyMuPDF 并行: {n_workers} 个 worker")
        failed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_pymupdf_one, f): f for f in pdf_files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="PDF→TXT"):
                if not fut.result():
                    failed += 1
        if failed:
            print(f"失败/跳过: {failed} 个")
    else:
        for pdf_file in tqdm(pdf_files, desc="PDF→TXT"):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            txt_path = os.path.join(OUT_DIR, pdf_file.replace(".pdf", ".txt"))
            ok = False
            if use_mineru:
                ok = extract_mineru(pdf_path, txt_path, MINERU_TEMP_BASE)
            if not ok and (METHOD == "pymupdf" or METHOD == "mineru_then_pymupdf"):
                ok = extract_pymupdf(pdf_path, txt_path)
            if not ok:
                print(f"跳过: {pdf_file}")

    print(f"输出目录: {OUT_DIR}")
    if use_mineru and os.path.isdir(MINERU_TEMP_BASE):
        print(f"MinerU 临时目录可手动删除: {MINERU_TEMP_BASE}")


if __name__ == "__main__":
    main()
