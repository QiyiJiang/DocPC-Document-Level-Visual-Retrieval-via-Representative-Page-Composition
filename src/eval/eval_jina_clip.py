#!/usr/bin/env python3
"""
Jina-CLIP 本地模型评估脚本。

默认按单向量余弦相似度排序，适用于 Jina-CLIP 这类 image/text 单向量模型。
推荐模型：jinaai/jina-clip-v2
"""

import os

# 强制离线：本地模型不再访问 Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import math
import random
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


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

        gt_entry = next((item for item in ground_truth if item["query"] == query), None)
        if not gt_entry:
            continue

        gt_items = gt_entry.get(pos_target_column, [])
        if isinstance(gt_items, str):
            gt_items = [gt_items]
        gt_names = {os.path.basename(item).rsplit(".", 1)[0] for item in gt_items}

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

            precision = num_hits / k if k > 0 else 0.0
            recall = num_hits / len(gt_names) if gt_names else 0.0

            mrr = 0.0
            for rank, name in enumerate(top_k):
                if is_hit(name):
                    mrr = 1 / (rank + 1)
                    break

            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gt_names), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

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


class JinaClipService:
    def __init__(self, model_path: str, truncate_dim: Optional[int] = 512):
        print("================================================")
        print(f"从 {model_path} 加载 Jina-CLIP 模型")
        print(f"设备: {DEVICE}")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("已使用 local_files_only 从本地加载，未访问网络")
        if hasattr(self.model, "to"):
            self.model = self.model.to(DEVICE)
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        self.truncate_dim = truncate_dim

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        try:
            return normalize_embeddings(
                self.model.encode_image(image_paths, truncate_dim=self.truncate_dim)
            )
        except TypeError:
            # 兼容 jina-clip-v1
            return normalize_embeddings(self.model.encode_image(image_paths))

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        try:
            return normalize_embeddings(
                self.model.encode_text(
                    texts,
                    task="retrieval.query",
                    truncate_dim=self.truncate_dim,
                )
            )
        except TypeError:
            # 兼容 jina-clip-v1
            return normalize_embeddings(self.model.encode_text(texts))


def batched_encode_images(
    service: JinaClipService,
    image_paths: List[str],
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图片"):
        batch = image_paths[i : i + batch_size]
        outputs.append(service.encode_images(batch))
    return np.vstack(outputs)


def batched_encode_texts(
    service: JinaClipService,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="编码查询"):
        batch = texts[i : i + batch_size]
        outputs.append(service.encode_texts(batch))
    return np.vstack(outputs)


def evaluate_jina_clip(
    model_path: str,
    image_dir: str,
    eval_dataset_path: str,
    results_dir: str,
    pos_target_column: str = "pos_target_for_deepseek",
    run_tag: str = "",
    batch_size: int = 32,
    truncate_dim: Optional[int] = 512,
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
                path = os.path.join(image_dir, os.path.basename(name))
                if os.path.exists(path):
                    image_set.add(path)
        image_paths = sorted(image_set)
        print(f"正例池模式: 找到 {len(image_paths)} 个唯一图片")
    if not image_paths:
        print("没有找到任何图片，跳过评估")
        return {}

    service = JinaClipService(model_path=model_path, truncate_dim=truncate_dim)

    print("\n步骤1: 编码图片...")
    image_vecs = batched_encode_images(service, image_paths, batch_size=batch_size)
    image_basenames = [os.path.basename(p).rsplit(".", 1)[0] for p in image_paths]
    print(f"图片向量形状: {image_vecs.shape}")

    query_indices = []
    queries = []
    for idx, item in enumerate(data):
        q = item.get("query", "")
        if q:
            query_indices.append(idx)
            queries.append(q)

    print(f"\n步骤2: 编码 {len(queries)} 条查询...")
    query_vecs = batched_encode_texts(service, queries, batch_size=batch_size)
    print(f"查询向量形状: {query_vecs.shape}")

    print("\n步骤3: 计算余弦相似度并排序...")
    sims = query_vecs @ image_vecs.T

    for i, data_idx in enumerate(query_indices):
        scores = sims[i]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        data[data_idx]["results"] = [image_basenames[j] for j in top_indices]

    if run_tag:
        results_path = os.path.join(results_dir, f"eval_{run_tag}.json")
        metrics_path = os.path.join(results_dir, f"metrics_{run_tag}.json")
    else:
        rid = random.randint(1, 1000000)
        results_path = os.path.join(results_dir, f"eval_jina_clip_{rid}.json")
        metrics_path = os.path.join(results_dir, f"metrics_jina_clip_{rid}.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"搜索结果保存到: {results_path}")

    print("\n步骤4: 计算评估指标...")
    ground_truth = load_json(eval_dataset_path)
    metrics = compute_metrics(data, ground_truth, [1, 3, 5, 10], pos_target_column)

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
    parser = argparse.ArgumentParser(description="Jina-CLIP 本地模型评估脚本")
    parser.add_argument("--model-path", type=str, required=True, help="本地 Jina-CLIP 模型目录")
    parser.add_argument("--image-dir", type=str, required=True, help="图片目录")
    parser.add_argument("--eval-dataset-path", type=str, required=True, help="查询 JSON 文件")
    parser.add_argument("--results-dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--pos-target-column", type=str, default="pos_target_for_deepseek")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--truncate-dim", type=int, default=512)
    parser.add_argument("--full-pool", action="store_true",
                        help="索引 image_dir 下全部图片（全库检索）；默认只索引正例文档")
    args = parser.parse_args()

    evaluate_jina_clip(
        model_path=args.model_path,
        image_dir=args.image_dir,
        eval_dataset_path=args.eval_dataset_path,
        results_dir=args.results_dir,
        pos_target_column=args.pos_target_column,
        run_tag=args.run_tag,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
        full_pool=args.full_pool,
    )


if __name__ == "__main__":
    main()
