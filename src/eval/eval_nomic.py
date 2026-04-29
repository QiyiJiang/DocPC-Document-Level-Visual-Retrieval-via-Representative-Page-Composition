#!/usr/bin/env python3
"""
Nomic 嵌入模型评估脚本。

使用 nomic-embed-vision-v1.5 编码图片，nomic-embed-text-v1.5 编码查询文本。
单向量余弦相似度排序，与 ColPali/ColQwen 的多向量 MaxSim 不同。

用法:
  python eval_nomic.py \
    --image-dir /path/to/images \
    --eval-dataset-path /path/to/query.json \
    --results-dir ./eval_results/nomic \
    --pos-target-column pos_target_for_deepseek \
    --run-tag "nomic__first4__biology"
"""
import os
import json
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from nomic import embed


TOP_K = 10


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(
    retrieved: List[Dict],
    ground_truth: List[Dict],
    k_values: List[int],
    pos_target_column: str = "pos_target",
) -> Dict:
    results_total = {
        k: {"precision": [], "recall": [], "mrr": [], "ndcg": []} for k in k_values
    }

    for query_entry in retrieved:
        query = query_entry["query"]
        predicted = query_entry.get("results", [])

        gt_entry = next(
            (item for item in ground_truth if item["query"] == query), None
        )
        if not gt_entry:
            continue

        gt_items = gt_entry.get(pos_target_column, [])
        if isinstance(gt_items, str):
            gt_items = [gt_items]
        gt_names = set()
        for item in gt_items:
            gt_names.add(os.path.basename(item).rsplit(".", 1)[0])

        unique_predicted = []
        seen = set()
        for item in predicted:
            if item not in seen:
                unique_predicted.append(item)
                seen.add(item)

        def is_hit(pred_name: str) -> bool:
            return os.path.basename(str(pred_name)).rsplit(".", 1)[0] in gt_names

        for k in k_values:
            top_k = unique_predicted[:k]
            hits = [i for i, name in enumerate(top_k) if is_hit(name)]
            num_hits = len(hits)
            dcg = sum(1 / math.log2(rank + 2) for rank in hits)

            precision = num_hits / k if k > 0 else 0
            recall = num_hits / len(gt_names) if gt_names else 0

            mrr = 0
            for rank, name in enumerate(top_k):
                if is_hit(name):
                    mrr = 1 / (rank + 1)
                    break

            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gt_names), k)))
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
    return avg


def embed_images(
    image_paths: List[str],
    model: str = "nomic-embed-vision-v1.5",
    batch_size: int = 50,
) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图片"):
        batch = image_paths[i : i + batch_size]
        output = embed.image(images=batch, model=model)
        all_embeddings.extend(output["embeddings"])
    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


def embed_texts(
    texts: List[str],
    model: str = "nomic-embed-text-v1.5",
    batch_size: int = 100,
) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="编码查询"):
        batch = texts[i : i + batch_size]
        output = embed.text(texts=batch, model=model, task_type="search_query")
        all_embeddings.extend(output["embeddings"])
    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms


def evaluate_nomic(
    image_dir: str,
    eval_dataset_path: str,
    results_dir: str,
    pos_target_column: str = "pos_target_for_deepseek",
    run_tag: str = "",
    vision_model: str = "nomic-embed-vision-v1.5",
    text_model: str = "nomic-embed-text-v1.5",
    batch_size: int = 50,
    full_pool: bool = False,
):
    os.makedirs(results_dir, exist_ok=True)
    data = load_json(eval_dataset_path)
    print(f"加载 {len(data)} 条查询数据")

    if full_pool:
        image_paths = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_paths.append(os.path.join(image_dir, f))
        print(f"全库模式: 找到 {len(image_paths)} 个图片")
    else:
        image_set = set()
        for item in data:
            names = item.get(pos_target_column, [])
            if isinstance(names, str):
                names = [names]
            for name in names:
                doc_key = os.path.basename(name)
                path = os.path.join(image_dir, doc_key)
                if os.path.exists(path):
                    image_set.add(path)
        image_paths = sorted(image_set)
        print(f"正例池模式: 找到 {len(image_paths)} 个唯一图片")

    if not image_paths:
        print("没有找到任何图片，跳过评估")
        return {}

    # ---- 步骤1: 编码图片 ----
    print("\n步骤1: 编码图片...")
    image_vecs = embed_images(image_paths, model=vision_model, batch_size=batch_size)
    image_basenames = [os.path.basename(p).rsplit(".", 1)[0] for p in image_paths]
    print(f"图片向量形状: {image_vecs.shape}")

    # ---- 步骤2: 编码查询 ----
    query_indices = []
    queries = []
    for idx, item in enumerate(data):
        q = item.get("query", "")
        if q:
            query_indices.append(idx)
            queries.append(q)

    print(f"\n步骤2: 编码 {len(queries)} 条查询...")
    query_vecs = embed_texts(queries, model=text_model, batch_size=batch_size)
    print(f"查询向量形状: {query_vecs.shape}")

    # ---- 步骤3: 计算相似度并排序 ----
    print("\n步骤3: 计算余弦相似度并排序...")
    sims = query_vecs @ image_vecs.T  # (num_queries, num_images)

    for i, data_idx in enumerate(query_indices):
        scores = sims[i]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        data[data_idx]["results"] = [image_basenames[j] for j in top_indices]

    # ---- 保存结果 ----
    if run_tag:
        results_path = os.path.join(results_dir, f"eval_{run_tag}.json")
        metrics_path = os.path.join(results_dir, f"metrics_{run_tag}.json")
    else:
        rid = random.randint(1, 1000000)
        results_path = os.path.join(results_dir, f"eval_nomic_{rid}.json")
        metrics_path = os.path.join(results_dir, f"metrics_nomic_{rid}.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"搜索结果保存到: {results_path}")

    # ---- 步骤4: 计算评估指标 ----
    print("\n步骤4: 计算评估指标...")
    k_values = [1, 3, 5, 10]
    ground_truth = load_json(eval_dataset_path)
    metrics = compute_metrics(data, ground_truth, k_values, pos_target_column)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n评估指标保存到: {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Nomic 嵌入模型评估脚本")
    parser.add_argument("--image-dir", type=str, required=True, help="图片目录")
    parser.add_argument("--eval-dataset-path", type=str, required=True, help="查询 JSON 文件")
    parser.add_argument("--results-dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--pos-target-column", type=str, default="pos_target_for_deepseek")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--vision-model", type=str, default="nomic-embed-vision-v1.5")
    parser.add_argument("--text-model", type=str, default="nomic-embed-text-v1.5")
    parser.add_argument("--batch-size", type=int, default=50, help="API 批次大小")
    parser.add_argument("--full-pool", action="store_true",
                        help="索引 image_dir 下全部图片（全库检索）；默认只索引正例文档")
    args = parser.parse_args()

    evaluate_nomic(
        image_dir=args.image_dir,
        eval_dataset_path=args.eval_dataset_path,
        results_dir=args.results_dir,
        pos_target_column=args.pos_target_column,
        run_tag=args.run_tag,
        vision_model=args.vision_model,
        text_model=args.text_model,
        batch_size=args.batch_size,
        full_pool=args.full_pool,
    )


if __name__ == "__main__":
    main()
