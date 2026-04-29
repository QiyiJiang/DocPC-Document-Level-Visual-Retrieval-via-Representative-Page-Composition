#!/usr/bin/env python3
"""
评估 BM25 和 BGE-M3 的检索结果（全量检索池）。
读取 500——eval_results/bm25-bge/ 下的 JSON 文件，
用 pos_target_for_deepseek 作为正例，
分别用 retrieved_docs_embedding（BGE-M3）和 retrieved_docs_bm25（BM25）作为预测结果，
计算 P@K, R@K, MRR@K, NDCG@K。

用法:
  python eval_bm25_bge.py \\
    --input-dir /data/docpc_project/500——eval_results/bm25-bge \\
    --output-dir /data/docpc_project/500——eval_results/bm25-bge
"""
import argparse
import json
import math
import os
import re
from typing import Dict, List


def to_stem(name: str) -> str:
    """去掉路径和任意后缀（.png / .txt 等），只保留文档 stem，便于统一比对。"""
    s = name.strip()
    if not s:
        return ""
    return os.path.basename(s).rsplit(".", 1)[0]


def has_non_empty_pos_target(item: Dict, pos_target_column: str) -> bool:
    """判断该条目的 pos_target_for_deepseek 是否非空（排除空串）。"""
    v = item.get(pos_target_column, [])
    if isinstance(v, str):
        v = [v]
    return bool(v and any(str(x).strip() for x in v))


def compute_metrics(
    data: List[Dict],
    pos_target_column: str,
    retrieved_column: str,
    k_values: List[int],
) -> Dict:
    # 先排除 pos_target_for_deepseek 为空的数据，只对有效正例的条目做评估
    data_filtered = [item for item in data if has_non_empty_pos_target(item, pos_target_column)]
    skipped = len(data) - len(data_filtered)
    data = data_filtered

    results_total = {k: {"precision": [], "recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for item in data:
        gt_items = item.get(pos_target_column, [])
        if isinstance(gt_items, str):
            gt_items = [gt_items]
        gt_items = [g for g in gt_items if str(g).strip()]
        if not gt_items:
            continue

        # 统一用 stem 比对，不受 .png / .txt 等后缀影响
        gt_names = {to_stem(g) for g in gt_items}

        predicted = item.get(retrieved_column, [])
        if isinstance(predicted, str):
            predicted = [predicted]

        unique_predicted = []
        seen = set()
        for p in predicted:
            base = to_stem(p)
            if base and base not in seen:
                unique_predicted.append(base)
                seen.add(base)

        for k in k_values:
            top_k = unique_predicted[:k]
            hits = [i for i, name in enumerate(top_k) if name in gt_names]
            num_hits = len(hits)

            precision = num_hits / k if k > 0 else 0
            recall = num_hits / len(gt_names) if gt_names else 0

            mrr = 0.0
            for rank, name in enumerate(top_k):
                if name in gt_names:
                    mrr = 1.0 / (rank + 1)
                    break

            dcg = sum(1.0 / math.log2(rank + 2) for rank in hits)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_names), k)))
            ndcg = dcg / idcg if idcg > 0 else 0

            results_total[k]["precision"].append(precision)
            results_total[k]["recall"].append(recall)
            results_total[k]["mrr"].append(mrr)
            results_total[k]["ndcg"].append(ndcg)

    avg = {}
    for k in k_values:
        n = len(results_total[k]["precision"])
        if n > 0:
            avg[f"P@{k}"] = round(sum(results_total[k]["precision"]) / n, 4)
            avg[f"R@{k}"] = round(sum(results_total[k]["recall"]) / n, 4)
            avg[f"MRR@{k}"] = round(sum(results_total[k]["mrr"]) / n, 4)
            avg[f"NDCG@{k}"] = round(sum(results_total[k]["ndcg"]) / n, 4)
        else:
            avg[f"P@{k}"] = 0.0
            avg[f"R@{k}"] = 0.0
            avg[f"MRR@{k}"] = 0.0
            avg[f"NDCG@{k}"] = 0.0
    avg["num_evaluated"] = len(results_total[k_values[0]]["precision"])
    avg["num_skipped_empty_gt"] = skipped
    return avg


def extract_category(filename: str) -> str:
    m = re.search(r"query_list_text_(\w+?)_with_pos_target", filename)
    return m.group(1) if m else os.path.splitext(filename)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/data/docpc_project/500——eval_results/bm25-bge",
                        help="全量检索池的 retrieved JSON 所在目录")
    parser.add_argument("--output-dir", type=str, default="/data/docpc_project/500——eval_results/bm25-bge")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k_values = [1, 3, 5, 10]

    json_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json") and "retrieved" in f)
    if not json_files:
        print(f"在 {args.input_dir} 下未找到 retrieved JSON 文件")
        return

    all_bge_metrics = {}
    all_bm25_metrics = {}

    for fname in json_files:
        fpath = os.path.join(args.input_dir, fname)
        cat = extract_category(fname)

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"类别: {cat}  ({len(data)} 条 query)")
        print(f"{'='*60}")

        bge_metrics = compute_metrics(data, "pos_target_for_deepseek", "retrieved_docs_embedding", k_values)
        bm25_metrics = compute_metrics(data, "pos_target_for_deepseek", "retrieved_docs_bm25", k_values)

        print(f"\n  BGE-M3 (retrieved_docs_embedding):")
        for m, v in bge_metrics.items():
            print(f"    {m}: {v}")

        print(f"\n  BM25 (retrieved_docs_bm25):")
        for m, v in bm25_metrics.items():
            print(f"    {m}: {v}")

        all_bge_metrics[cat] = bge_metrics
        all_bm25_metrics[cat] = bm25_metrics

    # 汇总所有类别的平均
    print(f"\n{'='*60}")
    print("所有类别汇总（各类别指标的平均）")
    print(f"{'='*60}")

    def avg_across_cats(all_metrics: Dict[str, Dict]) -> Dict:
        keys = [k for k in list(all_metrics.values())[0].keys() if k.startswith(("P@", "R@", "MRR@", "NDCG@"))]
        avg = {}
        for k in keys:
            vals = [m[k] for m in all_metrics.values()]
            avg[k] = round(sum(vals) / len(vals), 4)
        avg["num_categories"] = len(all_metrics)
        return avg

    bge_avg = avg_across_cats(all_bge_metrics)
    bm25_avg = avg_across_cats(all_bm25_metrics)

    print(f"\n  BGE-M3 平均:")
    for m, v in bge_avg.items():
        print(f"    {m}: {v}")

    print(f"\n  BM25 平均:")
    for m, v in bm25_avg.items():
        print(f"    {m}: {v}")

    # 保存结果
    results = {
        "per_category": {
            cat: {"bge_m3": all_bge_metrics[cat], "bm25": all_bm25_metrics[cat]}
            for cat in sorted(all_bge_metrics.keys())
        },
        "average": {"bge_m3": bge_avg, "bm25": bm25_avg},
    }
    out_path = os.path.join(args.output_dir, "bm25_bge_metrics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n汇总结果已保存: {out_path}")


if __name__ == "__main__":
    main()
