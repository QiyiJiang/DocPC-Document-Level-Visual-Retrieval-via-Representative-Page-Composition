#!/usr/bin/env python3
"""
规模梯度实验：按不同比例采样正例文档，观察评估指标随规模变化的趋势。

核心逻辑：
1. 固定种子打乱所有正例文档，按比例取嵌套子集（20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100%）
2. 在最小比例（如 20%）确定 query 集合：只排除 pos_target 全部不在池中的 query；
   部分在的保留，去掉不在的 target
3. 后续比例（40%/60%/80%/100%）query 集合不变，只随着文档池扩大恢复之前被去掉的 target
4. 一次性编码全部图片和 query，纯内存 MaxSim 搜索
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_colqwen25_model import ColQwenService, load_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOPK = 10


def get_targets(item: Dict, col: str) -> List[str]:
    """统一返回 pos_target 列表"""
    val = item.get(col, "")
    if isinstance(val, str):
        return [val] if val else []
    return val if val else []


def collect_pos_doc_keys(data: List[Dict], col: str) -> List[str]:
    """收集数据集中所有唯一的正例文档 key（原始 basename，如 0046587.png）"""
    docs = set()
    for item in data:
        for t in get_targets(item, col):
            key = os.path.basename(t)
            if key:
                docs.add(key)
    return sorted(docs)


def maxsim_search(
    query_emb: np.ndarray,
    doc_embeddings: Dict[str, np.ndarray],
    top_k: int = TOPK,
) -> List[str]:
    """纯内存 MaxSim 搜索，返回 top_k 文档 base_name"""
    q = query_emb.astype(np.float32)
    scores = []
    for doc_name, doc_emb in doc_embeddings.items():
        sim = q @ doc_emb.astype(np.float32).T
        scores.append((doc_name, float(sim.max(axis=1).sum())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]


def compute_metrics_from_lists(
    search_results: List[Dict],
    ground_truth: List[Dict],
    k_values: List[int],
    pos_target_column: str,
) -> Dict:
    """计算 P@K, R@K, MRR@K, NDCG@K"""
    results_total = {k: {"precision": [], "recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for sr in search_results:
        query = sr["query"]
        predicted = sr.get("results", [])

        gt_entry = next((g for g in ground_truth if g["query"] == query), None)
        if not gt_entry:
            continue

        gt_items = gt_entry.get(pos_target_column, [])
        if isinstance(gt_items, str):
            gt_items = [gt_items]
        gt_names = {os.path.basename(g).rsplit(".", 1)[0] for g in gt_items}
        if not gt_names:
            continue

        unique_predicted = list(dict.fromkeys(predicted))

        def is_hit(pred):
            return os.path.basename(str(pred)).rsplit(".", 1)[0] in gt_names

        for k in k_values:
            top_k = unique_predicted[:k]
            hits = [i for i, name in enumerate(top_k) if is_hit(name)]
            num_hits = len(hits)
            precision = num_hits / k if k > 0 else 0
            recall = num_hits / len(gt_names) if gt_names else 0
            mrr = 0.0
            for rank, name in enumerate(top_k):
                if is_hit(name):
                    mrr = 1.0 / (rank + 1)
                    break
            dcg = sum(1.0 / math.log2(r + 2) for r in hits)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_names), k)))
            ndcg = dcg / idcg if idcg > 0 else 0

            results_total[k]["precision"].append(precision)
            results_total[k]["recall"].append(recall)
            results_total[k]["mrr"].append(mrr)
            results_total[k]["ndcg"].append(ndcg)

    avg = {}
    for k in k_values:
        for metric_name in ["precision", "recall", "mrr", "ndcg"]:
            vals = results_total[k][metric_name]
            label = {"precision": "P", "recall": "R", "mrr": "MRR", "ndcg": "NDCG"}[metric_name]
            avg[f"{label}@{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return avg


def main():
    parser = argparse.ArgumentParser(description="规模梯度实验")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--eval-dataset-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--pos-target-column", type=str, default="pos_target_for_deepseek")
    parser.add_argument("--sample-ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help="采样比例列表，如 0.2 0.4 0.6 0.8 1.0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确保基线和微调模型采样相同子集）")
    parser.add_argument("--run-tag", type=str, default="", help="标签（如类别名）")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU 编号")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.results_dir, exist_ok=True)

    full_data = load_json(args.eval_dataset_path)
    all_doc_keys = collect_pos_doc_keys(full_data, args.pos_target_column)
    category = args.run_tag or os.path.basename(os.path.dirname(args.eval_dataset_path))

    print(f"数据集: {args.eval_dataset_path}")
    print(f"全部 query: {len(full_data)}, 全部正例文档: {len(all_doc_keys)}")
    print(f"采样比例: {args.sample_ratios}, 种子: {args.seed}")

    # 固定种子打乱文档（嵌套子集：20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100%）
    rng = random.Random(args.seed)
    shuffled_keys = list(all_doc_keys)
    rng.shuffle(shuffled_keys)

    ratios = sorted(args.sample_ratios)

    # ────── 在最小比例确定 query 集合 ──────
    min_ratio = ratios[0]
    n_min = max(1, int(len(shuffled_keys) * min_ratio))
    min_pool = set(shuffled_keys[:n_min])

    selected_queries = []
    for item in full_data:
        targets = get_targets(item, args.pos_target_column)
        if not targets:
            continue
        keys = {os.path.basename(t) for t in targets}
        if keys & min_pool:
            selected_queries.append(item)

    print(f"\n最小比例 {int(min_ratio * 100)}%: 文档池 {len(min_pool)}, "
          f"选中 query {len(selected_queries)}/{len(full_data)}")

    # ────── 加载模型（仅一次） ──────
    colqwen_service = ColQwenService(model_path=args.model_path)

    # ────── 一次性编码全部图片 ──────
    print(f"\n编码全部 {len(all_doc_keys)} 张图片...")
    all_img_embs: Dict[str, np.ndarray] = {}
    for doc_key in tqdm(all_doc_keys, desc="编码图片"):
        img_path = os.path.join(args.image_dir, doc_key)
        if not os.path.exists(img_path):
            continue
        base_name = doc_key.rsplit(".", 1)[0]
        try:
            emb = colqwen_service.multi_vectorize_image(Image.open(img_path))
            all_img_embs[base_name] = emb
        except Exception as e:
            print(f"  编码 {doc_key} 失败: {e}")
    print(f"成功编码 {len(all_img_embs)} 张图片")

    # ────── 一次性编码选中的 query ──────
    unique_queries = list({item.get("query", "") for item in selected_queries if item.get("query")})
    print(f"\n编码 {len(unique_queries)} 条 query...")
    all_query_embs: Dict[str, np.ndarray] = {}
    for q in tqdm(unique_queries, desc="编码 query"):
        try:
            all_query_embs[q] = colqwen_service.multi_vectorize_text(q)
        except Exception as e:
            print(f"  编码 query 失败: {e}")
    print(f"成功编码 {len(all_query_embs)} 条 query")

    # ────── 逐比例评估 ──────
    all_results = []
    k_values = [1, 3, 5, 10]

    for ratio in ratios:
        n_sample = max(1, int(len(shuffled_keys) * ratio))
        if ratio >= 1.0:
            n_sample = len(shuffled_keys)
        current_pool = set(shuffled_keys[:n_sample])
        ratio_pct = int(ratio * 100)

        # 对固定的 query 集合，根据当前文档池调整 pos_target
        adjusted_data = []
        total_targets = 0
        for item in selected_queries:
            targets = get_targets(item, args.pos_target_column)
            valid = [t for t in targets if os.path.basename(t) in current_pool]
            if valid:
                new_item = dict(item)
                new_item[args.pos_target_column] = valid
                adjusted_data.append(new_item)
                total_targets += len(valid)

        print(f"\n{'=' * 60}")
        print(f"比例 {ratio_pct}%: 文档池 {len(current_pool)}/{len(all_doc_keys)}, "
              f"query {len(adjusted_data)}, 正例总数 {total_targets}")
        print(f"{'=' * 60}")

        if not adjusted_data:
            print("  无有效 query，跳过")
            continue

        sampled_bases = {k.rsplit(".", 1)[0] for k in current_pool}
        doc_pool_embs = {b: all_img_embs[b] for b in sampled_bases if b in all_img_embs}

        if not doc_pool_embs:
            print("  无可用图片嵌入，跳过")
            continue

        search_results = []
        for item in tqdm(adjusted_data, desc=f"[{ratio_pct}%] 搜索"):
            query = item.get("query", "")
            if not query or query not in all_query_embs:
                continue
            results = maxsim_search(all_query_embs[query], doc_pool_embs, top_k=TOPK)
            search_results.append({"query": query, "results": results})

        metrics = compute_metrics_from_lists(
            search_results, adjusted_data, k_values, args.pos_target_column
        )

        result_info = {
            "ratio": ratio,
            "ratio_pct": ratio_pct,
            "total_docs": len(all_doc_keys),
            "sampled_docs": len(current_pool),
            "selected_queries": len(selected_queries),
            "valid_queries": len(adjusted_data),
            "total_pos_targets": total_targets,
            "metrics": metrics,
        }
        all_results.append(result_info)

        ratio_tag = f"{category}_{ratio_pct}pct"
        metrics_path = os.path.join(args.results_dir, f"metrics_{ratio_tag}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result_info, f, indent=4, ensure_ascii=False)

        print(f"  结果: ", end="")
        for k, v in metrics.items():
            print(f"{k}={v}  ", end="")
        print()

    # ────── 汇总 ──────
    print(f"\n{'=' * 100}")
    print(f"规模梯度实验汇总 — {category}  (固定 query 集: {len(selected_queries)} 条)")
    print(f"{'=' * 100}")
    header = f"{'比例':>6} {'文档数':>8} {'Query数':>8} {'正例数':>8}"
    for k in k_values:
        header += f" {'P@'+str(k):>7} {'R@'+str(k):>7} {'NDCG@'+str(k):>8}"
    print(header)
    print("-" * 100)
    for r in all_results:
        m = r["metrics"]
        line = (f"{r['ratio_pct']:>5}% {r['sampled_docs']:>8} "
                f"{r['valid_queries']:>8} {r['total_pos_targets']:>8}")
        for k in k_values:
            line += (f" {m.get(f'P@{k}', 0):>7.4f} {m.get(f'R@{k}', 0):>7.4f} "
                     f"{m.get(f'NDCG@{k}', 0):>8.4f}")
        print(line)

    summary_path = os.path.join(args.results_dir, f"scale_summary_{category}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n汇总保存到: {summary_path}")


if __name__ == "__main__":
    main()
