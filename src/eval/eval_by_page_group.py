#!/usr/bin/env python3
"""按文档页面数量分组，基于已有全量检索结果分析召回效果。

思路：
  1. 从 {cat}_pos_target_for_deepseek/ 读 PDF 统计页数，分三组（≤10 / 11~20 / >20）
  2. 从 data_text_with_{cat}_cluster_ids_with_querys_for_deepseek.json 得到
     每个正例文档属于哪些 query
  3. 从已有检索结果读取（支持 eval_xxx__strategy__{cat}.json 或 reference_test_datasets_*-{cat}.json）
  4. 按分组切片计算指标，索引规模不变，比较公平

使用方式:
  cd /data/docpc_project
  python eval_by_page_group.py \
      --retrieval-dir /data/docpc_project/500——eval_results/combo \
      --eval-name eval_colqwen_pdfa_all_first4__first4 \
      --output-dir /data/docpc_project/500——eval_results/by_page_group
"""

import os
import json
import math
import glob
import argparse

import fitz
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Set, Tuple

CATEGORIES = [
    "biology", "education", "finance", "government",
    "industrial", "legal", "research",
]

PDF_ROOT = "/data/docpc_project/datasets/pdfa_test/pages_4_30"
QUERY_ROOT = "/data/docpc_project/dataset_generate_new/pdfa_test"

GROUP_DEFS = [
    ("pages_le_10",    "≤10 页",  lambda p: p <= 10),
    ("pages_11_to_20", "11~20 页", lambda p: 10 < p <= 20),
    ("pages_gt_20",    ">20 页",  lambda p: p > 20),
]


# ---------------------------------------------------------------------------
# 1. PDF 页数统计 & 分组
# ---------------------------------------------------------------------------

def scan_pos_target_pdfs() -> Dict[str, Dict]:
    """扫描 {cat}_pos_target_for_deepseek 下的 PDF，统计页数。

    Returns:
        { "biology/0046587": {"cat": "biology", "stem": "0046587", "pages": 5}, ... }
    """
    all_docs: Dict[str, Dict] = {}
    for cat in CATEGORIES:
        pdf_dir = os.path.join(PDF_ROOT, f"{cat}_pos_target_for_deepseek")
        if not os.path.isdir(pdf_dir):
            print(f"[WARN] 不存在: {pdf_dir}")
            continue
        fnames = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        for fname in tqdm(fnames, desc=f"扫描 {cat} 正例 PDF"):
            stem = fname.rsplit(".", 1)[0]
            doc = fitz.open(os.path.join(pdf_dir, fname))
            n_pages = len(doc)
            doc.close()
            all_docs[f"{cat}/{stem}"] = {
                "cat": cat, "stem": stem, "pages": n_pages,
            }
    return all_docs


def group_docs(all_docs: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    groups = {name: {} for name, _, _ in GROUP_DEFS}
    for doc_key, info in all_docs.items():
        for gname, _, pred in GROUP_DEFS:
            if pred(info["pages"]):
                groups[gname][doc_key] = info
                break
    return groups


# ---------------------------------------------------------------------------
# 2. 文档 → Query 映射
# ---------------------------------------------------------------------------

def build_doc_to_queries() -> Dict[str, List[str]]:
    """从 data_text_with_{cat}_cluster_ids_with_querys_for_deepseek.json 构建映射。

    Returns:
        { "biology/0046587": ["query text 1", "query text 2", ...], ... }
    """
    mapping: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        fpath = os.path.join(
            QUERY_ROOT, cat,
            f"data_text_with_{cat}_cluster_ids_with_querys_for_deepseek.json",
        )
        if not os.path.exists(fpath):
            print(f"[WARN] 不存在: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            qs = entry.get("querys_for_deepseek", [])
            if not qs:
                continue
            stem = entry["text_name"].rsplit(".", 1)[0]
            key = f"{cat}/{stem}"
            mapping[key] = qs
    return mapping


# ---------------------------------------------------------------------------
# 3. 加载已有检索结果
# ---------------------------------------------------------------------------

def load_retrieval_results(
    retrieval_dir: str,
    eval_name: str = None,
) -> Dict[str, Dict[str, dict]]:
    """加载检索结果。

    若指定 eval_name（如 eval_colqwen_pdfa_all_first4__first4），则加载
    {retrieval_dir}/{eval_name}__{cat}.json；
    否则按旧格式加载 reference_test_datasets_*-{cat}.json。

    Returns:
        { "biology": { "query text": {"results": [...], ...}, ... }, ... }
    """
    results_by_cat: Dict[str, Dict[str, dict]] = {}

    for cat in CATEGORIES:
        if eval_name:
            fpath = os.path.join(retrieval_dir, f"{eval_name}__{cat}.json")
            if not os.path.isfile(fpath):
                print(f"[WARN] {cat} 的检索结果文件不存在: {fpath}")
                continue
            matches = [fpath]
        else:
            pattern = os.path.join(retrieval_dir, f"reference_test_datasets_*-{cat}.json")
            matches = glob.glob(pattern)
        if not matches:
            print(f"[WARN] {cat} 的检索结果文件不存在")
            continue
        fpath = matches[0]
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        cat_map: Dict[str, dict] = {}
        for entry in data:
            if "query" not in entry or "results" not in entry:
                continue
            cat_map[entry["query"]] = entry
        results_by_cat[cat] = cat_map
        print(f"  [{cat}] 加载 {len(cat_map)} 条检索结果 ← {os.path.basename(fpath)}")

    return results_by_cat


# ---------------------------------------------------------------------------
# 4. 指标计算
# ---------------------------------------------------------------------------

def compute_metrics(
    eval_entries: List[dict],
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    eval_entries: [{"results": [stem, ...], "gt_stems": set(stem, ...)}, ...]
    """
    buckets = {k: {"P": [], "R": [], "MRR": [], "NDCG": []} for k in k_values}

    for entry in eval_entries:
        predicted = list(dict.fromkeys(entry["results"]))
        gt_set: Set[str] = entry["gt_stems"]
        if not gt_set:
            continue

        for k in k_values:
            top_k = predicted[:k]
            hits = [i for i, name in enumerate(top_k) if name in gt_set]
            n_hits = len(hits)

            buckets[k]["P"].append(n_hits / k)
            buckets[k]["R"].append(n_hits / len(gt_set))

            mrr = 0.0
            for rank, name in enumerate(top_k):
                if name in gt_set:
                    mrr = 1.0 / (rank + 1)
                    break
            buckets[k]["MRR"].append(mrr)

            dcg = sum(1.0 / math.log2(r + 2) for r in hits)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), k)))
            buckets[k]["NDCG"].append(dcg / idcg if idcg > 0 else 0.0)

    avg = {}
    for k in k_values:
        for m in ("P", "R", "MRR", "NDCG"):
            vals = buckets[k][m]
            avg[f"{m}@{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return avg


# ---------------------------------------------------------------------------
# 5. 主逻辑
# ---------------------------------------------------------------------------

def analyze_group(
    group_name: str,
    group_docs: Dict[str, Dict],
    doc_to_queries: Dict[str, List[str]],
    results_by_cat: Dict[str, Dict[str, dict]],
) -> Tuple[List[dict], Dict]:
    """对一组文档，找出其关联 query，从已有结果中取召回列表，计算指标。"""

    group_stems_by_cat: Dict[str, Set[str]] = defaultdict(set)
    for _, info in group_docs.items():
        group_stems_by_cat[info["cat"]].add(info["stem"])

    # 收集每个 query 在本组的 ground truth stems
    # key = (cat, query_text)，防止跨类别 query 文本重复
    query_gt: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for doc_key, info in group_docs.items():
        cat, stem = info["cat"], info["stem"]
        qs = doc_to_queries.get(doc_key, [])
        for q in qs:
            query_gt[(cat, q)].add(stem)

    # 构建评估条目
    eval_entries: List[dict] = []
    missing_results = 0

    for (cat, query_text), gt_stems in query_gt.items():
        cat_results = results_by_cat.get(cat, {})
        result_entry = cat_results.get(query_text)
        if result_entry is None:
            missing_results += 1
            continue

        retrieved = result_entry.get("results", [])
        eval_entries.append({
            "query": query_text,
            "category": cat,
            "results": retrieved,
            "gt_stems": gt_stems,
        })

    if missing_results:
        print(f"  [{group_name}] {missing_results} 条 query 在检索结果中未找到")

    if not eval_entries:
        return eval_entries, {}

    metrics = compute_metrics(eval_entries)
    return eval_entries, metrics


def main():
    parser = argparse.ArgumentParser(
        description="按页面数量分组，基于已有检索结果分析召回效果",
    )
    parser.add_argument(
        "--retrieval-dir", type=str, required=True,
        help="已有检索结果目录",
    )
    parser.add_argument(
        "--eval-name", type=str, default=None,
        help="测评结果文件名前缀，如 eval_colqwen_pdfa_all_first4__first4；"
             "将加载 {retrieval_dir}/{eval_name}__{cat}.json。不填则按 reference_test_datasets_*-{cat}.json 查找",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/data/docpc_project/eval_results/by_page_group",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: 扫描正例 PDF 统计页数 ----
    print("Step 1: 扫描正例 PDF 统计页数...")
    all_docs = scan_pos_target_pdfs()
    groups = group_docs(all_docs)

    print(f"\n正例文档总数: {len(all_docs)}")
    for gname, label, _ in GROUP_DEFS:
        gdocs = groups[gname]
        if gdocs:
            pages = [d["pages"] for d in gdocs.values()]
            print(f"  {label} ({gname}): {len(gdocs)} 文档, "
                  f"页数 {min(pages)}~{max(pages)}")
        else:
            print(f"  {label} ({gname}): 0 文档")

    # ---- Step 2: 构建文档→Query 映射 ----
    print("\nStep 2: 构建文档→Query 映射...")
    doc_to_queries = build_doc_to_queries()
    total_pairs = sum(len(qs) for qs in doc_to_queries.values())
    print(f"  {len(doc_to_queries)} 个正例文档, 共 {total_pairs} 条 (文档, query) 对")

    # ---- Step 3: 加载已有检索结果 ----
    print(f"\nStep 3: 加载检索结果 ({args.retrieval_dir})...")
    if args.eval_name:
        print(f"  使用 eval 文件: {args.eval_name}__{{cat}}.json")
    results_by_cat = load_retrieval_results(args.retrieval_dir, eval_name=args.eval_name)

    # ---- Step 4: 逐组分析 ----
    print("\nStep 4: 逐组分析...")
    summary = {}

    for gname, label, _ in GROUP_DEFS:
        gdocs = groups[gname]
        print(f"\n{'=' * 60}")
        print(f"[{label}] {gname}: {len(gdocs)} 正例文档")
        print(f"{'=' * 60}")

        eval_entries, metrics = analyze_group(
            gname, gdocs, doc_to_queries, results_by_cat,
        )

        pages = [d["pages"] for d in gdocs.values()] if gdocs else []
        info = {
            "label": label,
            "num_pos_docs": len(gdocs),
            "page_range": f"{min(pages)}~{max(pages)}" if pages else "N/A",
            "num_queries": len(eval_entries),
        }

        if metrics:
            info.update(metrics)
            print(f"  有效 Query: {len(eval_entries)}")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print("  （无有效 query，跳过）")

        summary[gname] = info

        # 保存该组详细结果
        detail_path = os.path.join(args.output_dir, f"detail_{gname}.json")
        detail_out = []
        for e in eval_entries:
            detail_out.append({
                "query": e["query"],
                "category": e["category"],
                "gt_stems": sorted(e["gt_stems"]),
                "results": e["results"],
            })
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_out, f, ensure_ascii=False, indent=2)

    # ---- 汇总 ----
    print(f"\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")

    header = f"{'组别':<20} {'文档数':>6} {'页数范围':>10} {'Query数':>8}"
    for k in (1, 3, 5, 10):
        header += f" {'NDCG@'+str(k):>8} {'R@'+str(k):>8}"
    print(header)
    print("-" * len(header))

    for gname, label, _ in GROUP_DEFS:
        info = summary[gname]
        row = (f"{label:<20} {info['num_pos_docs']:>6} "
               f"{info['page_range']:>10} {info['num_queries']:>8}")
        for k in (1, 3, 5, 10):
            ndcg = info.get(f"NDCG@{k}", 0.0)
            recall = info.get(f"R@{k}", 0.0)
            row += f" {ndcg:>8.4f} {recall:>8.4f}"
        print(row)

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")

    # 保存页数分布
    page_dist = {}
    for doc_key, info in all_docs.items():
        page_dist[doc_key] = {
            "cat": info["cat"],
            "stem": info["stem"],
            "pages": info["pages"],
        }
    dist_path = os.path.join(args.output_dir, "page_distribution.json")
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(page_dist, f, ensure_ascii=False, indent=2)
    print(f"页数分布已保存: {dist_path}")


if __name__ == "__main__":
    main()
