#!/usr/bin/env python3
"""
页面级检索 + 策略聚合评估。

与拼接图评估不同，本脚本：
1. 对文档的每一页单独编码
2. 按策略（first4 / first2_last2 / random4 / uniform4 / last4）选择每个文档的4页
3. 分别计算 query 与这4页的 MaxSim，聚合为文档分数
4. 排序、计算 P/R/MRR/NDCG

用法:
  python eval_colpali_page_strategy.py \
    --model-path /path/to/colpali_model \
    --eval-dataset-path /path/to/query.json \
    --metadata-path /path/to/image_page_metadata.json \
    --image-dir /path/to/page_images \
    --results-dir ./eval_results \
    --strategy all \
    --score-agg max \
    --pos-target-column pos_target_for_deepseek
"""
import os
import sys
import math
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from typing import List, Dict, Optional, Set

DEVICES = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALL_STRATEGIES = ["first4", "first9", "first2_last2", "random4", "uniform4", "last4", "all_pages"]


# ────────────────────── 数据加载 ──────────────────────

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_doc_to_ordered_pages(metadata_path: str) -> Dict[str, List[str]]:
    """
    doc_base -> [page_base_0, page_base_1, ...] 按页码排序。
    page_base 格式如 "0046587_0"（不含扩展名），
    注意过滤掉 page_name == document_name 的整文档图条目。
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    doc_pages: Dict[str, List[tuple]] = {}
    for entry in meta:
        page_name = entry.get("page_name", "")
        document_name = entry.get("document_name", "")
        if not page_name or not document_name:
            continue
        page_base = os.path.basename(page_name).rsplit(".", 1)[0]
        doc_base = os.path.basename(document_name).rsplit(".", 1)[0]
        if page_base == doc_base:
            continue
        parts = page_base.rsplit("_", 1)
        page_idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        if doc_base not in doc_pages:
            doc_pages[doc_base] = []
        doc_pages[doc_base].append((page_idx, page_base))

    result = {}
    for doc_base, pages in doc_pages.items():
        pages.sort(key=lambda x: x[0])
        result[doc_base] = [p[1] for p in pages]
    return result


def load_doc_to_page_files(metadata_path: str) -> Dict[str, List[str]]:
    """doc_key (如 '0046587.png') -> [page_filename, ...] 用于收集需编码的图片路径。"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out: Dict[str, List[str]] = {}
    for entry in meta:
        page_name = entry.get("page_name", "")
        document_name = entry.get("document_name", "")
        if not document_name or not page_name:
            continue
        page_base = os.path.basename(page_name).rsplit(".", 1)[0]
        doc_base = os.path.basename(document_name).rsplit(".", 1)[0]
        if page_base == doc_base:
            continue
        doc_key = os.path.basename(document_name)
        if doc_key not in out:
            out[doc_key] = []
        out[doc_key].append(os.path.basename(page_name))
    return out


# ────────────────────── 策略选页 ──────────────────────

def select_page_indices(n_pages: int, strategy: str, doc_name: str = "") -> List[int]:
    """根据策略返回去重后的页面索引列表。"""
    if n_pages == 0:
        return []

    if strategy == "first4":
        indices = list(range(min(4, n_pages)))

    elif strategy == "first9":
        indices = list(range(min(9, n_pages)))

    elif strategy == "first2_last2":
        if n_pages >= 4:
            indices = [0, 1, n_pages - 2, n_pages - 1]
        elif n_pages == 3:
            indices = [0, 1, 2]
        else:
            indices = list(range(n_pages))

    elif strategy == "random4":
        rng = random.Random(hash(doc_name) & 0xFFFFFFFF)
        if n_pages >= 4:
            indices = sorted(rng.sample(range(n_pages), 4))
        else:
            indices = list(range(n_pages))

    elif strategy == "uniform4":
        if n_pages >= 4:
            indices = sorted(set(round(i * (n_pages - 1) / 3) for i in range(4)))
        else:
            indices = list(range(n_pages))

    elif strategy == "last4":
        if n_pages >= 4:
            indices = [n_pages - 4, n_pages - 3, n_pages - 2, n_pages - 1]
        else:
            indices = list(range(n_pages))

    elif strategy == "all_pages":
        indices = list(range(n_pages))

    else:
        raise ValueError(f"未知策略: {strategy}")

    return sorted(set(indices))


# ────────────────────── 指标计算 ──────────────────────

def compute_metrics(
    retrieved: List[Dict],
    ground_truth: List[Dict],
    k_values: List[int],
    pos_target_column: str = "pos_target",
) -> Dict:
    """P@K, R@K, MRR@K, NDCG@K（结果和正例均为 document basename）。"""
    results_total = {k: {"precision": [], "recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for query_entry in retrieved:
        query = query_entry["query"]
        predicted = query_entry.get("results", [])

        gt_entry = next((item for item in ground_truth if item["query"] == query), None)
        if not gt_entry:
            continue
        gt_items = gt_entry.get(pos_target_column, [])
        if isinstance(gt_items, str):
            gt_items = [gt_items]
        gt_names = {os.path.basename(g).rsplit(".", 1)[0] for g in gt_items}
        if not gt_names:
            continue

        unique_predicted = list(dict.fromkeys(predicted))

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

            dcg = sum(1.0 / math.log2(r + 2) for r in hits)
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
            for m in ("P", "R", "MRR", "NDCG"):
                avg[f"{m}@{k}"] = 0.0
    return avg


# ────────────────────── 模型服务 ──────────────────────

class ModelService:
    def __init__(self, model_path: str, model_type: str = "colpali") -> None:
        print(f"从 {model_path} 加载 {model_type} 模型")
        if model_type == "colpali":
            self.model = ColPali.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=DEVICES
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model_path, use_fast=True)
        else:
            self.model = ColQwen2_5.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=DEVICES
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(model_path, use_fast=True)

    def encode_image(self, img: Image.Image) -> np.ndarray:
        batch = self.processor.process_images([img]).to(self.model.device)
        with torch.no_grad():
            emb = self.model(**batch)
        return emb[0].cpu().float().numpy()

    def encode_text(self, query: str) -> np.ndarray:
        batch = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            emb = self.model(**batch)
        return emb[0].cpu().float().numpy()


# ────────────────────── 页面编码 ──────────────────────

def collect_and_encode_pages(
    eval_dataset_path: str,
    image_dir: str,
    metadata_path: str,
    pos_target_column: str,
    model_service: ModelService,
    full_pool: bool = False,
    pool_docs: Optional[Set[str]] = None,
) -> Dict[str, np.ndarray]:
    """收集待编码的页面图片并编码。返回 {page_base: embedding}。
    full_pool=False 且 pool_docs=None: 只编码正例文档的页面。
    full_pool=True 且 pool_docs=None: 编码 metadata 中全部文档的页面。
    pool_docs 非空: 只编码该文档集合的页面（先选文档再加载对应图片）。
    """
    doc_to_page_files = load_doc_to_page_files(metadata_path)

    if pool_docs is not None:
        image_files = set()
        for doc_key, page_list in doc_to_page_files.items():
            doc_base = os.path.basename(doc_key).rsplit(".", 1)[0]
            if doc_base in pool_docs:
                for page_file in page_list:
                    image_files.add(os.path.join(image_dir, page_file))
        print(f"部分池模式: 共 {len(pool_docs)} 个文档, 找到 {len(image_files)} 个页面图片待编码")
    elif full_pool:
        image_files = set()
        for doc_key, page_list in doc_to_page_files.items():
            for page_file in page_list:
                image_files.add(os.path.join(image_dir, page_file))
        print(f"全库模式: 找到 {len(image_files)} 个页面图片待编码")
    else:
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_files = set()
        for item in data:
            names = item.get(pos_target_column, "")
            if isinstance(names, str):
                names = [names]
            for name in names:
                doc_key = os.path.basename(name)
                for page_file in doc_to_page_files.get(doc_key, []):
                    image_files.add(os.path.join(image_dir, page_file))
        print(f"正例池模式: 找到 {len(image_files)} 个页面图片待编码")

    page_embeddings: Dict[str, np.ndarray] = {}
    for img_path in tqdm(sorted(image_files), desc="编码页面"):
        base = os.path.basename(img_path).rsplit(".", 1)[0]
        if base in page_embeddings:
            continue
        try:
            emb = model_service.encode_image(Image.open(img_path))
            page_embeddings[base] = emb
        except Exception as e:
            print(f"编码失败 {img_path}: {e}")

    print(f"成功编码 {len(page_embeddings)} 个页面")
    return page_embeddings


# ────────────────────── 策略搜索 ──────────────────────

def search_documents_strategy(
    query_emb: np.ndarray,
    page_embeddings: Dict[str, np.ndarray],
    doc_to_ordered_pages: Dict[str, List[str]],
    strategy: str,
    score_agg: str = "max",
    limit: int = 10,
) -> List[str]:
    """对每个文档按策略选页、计算 MaxSim、聚合并排序。返回 top-K 文档 basenames。"""
    q = np.array(query_emb, dtype=np.float32)
    if len(q.shape) == 1:
        q = q.reshape(1, -1)

    doc_scores: Dict[str, float] = {}
    for doc_base, pages in doc_to_ordered_pages.items():
        indices = select_page_indices(len(pages), strategy, doc_name=doc_base)
        selected = [pages[i] for i in indices]

        scores = []
        for page_base in selected:
            if page_base in page_embeddings:
                vecs = page_embeddings[page_base]
                if len(vecs.shape) == 1:
                    vecs = vecs.reshape(1, -1)
                sim = q @ vecs.T
                scores.append(float(sim.max(axis=1).sum()))

        if scores:
            doc_scores[doc_base] = max(scores) if score_agg == "max" else sum(scores) / len(scores)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:limit]]


# ────────────────────── 单策略评估 ──────────────────────

def run_evaluation(
    model_service: ModelService,
    page_embeddings: Dict[str, np.ndarray],
    doc_to_ordered_pages: Dict[str, List[str]],
    eval_data: List[Dict],
    strategy: str,
    score_agg: str,
    pos_target_column: str,
    results_dir: str,
    run_tag: str,
) -> Dict:
    k_values = [1, 3, 5, 10]
    retrieved = []

    for item in tqdm(eval_data, desc=f"搜索 [{strategy}]"):
        query = item.get("query", "")
        if not query:
            continue
        query_emb = model_service.encode_text(query)
        top_docs = search_documents_strategy(
            query_emb, page_embeddings, doc_to_ordered_pages,
            strategy, score_agg, limit=10,
        )
        retrieved.append({"query": query, "results": top_docs})

    metrics = compute_metrics(retrieved, eval_data, k_values, pos_target_column)

    if run_tag:
        results_path = os.path.join(results_dir, f"eval_{run_tag}__{strategy}.json")
        metrics_path = os.path.join(results_dir, f"metrics_{run_tag}__{strategy}.json")
    else:
        rand_id = random.randint(1, 1000000)
        results_path = os.path.join(results_dir, f"page_strategy_{strategy}_{rand_id}.json")
        metrics_path = os.path.join(results_dir, f"page_strategy_{strategy}_metrics_{rand_id}.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(retrieved, f, indent=4, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"\n策略 [{strategy}] 评估结果:")
    for m, v in metrics.items():
        print(f"  {m}: {v}")
    print(f"  结果: {results_path}")
    print(f"  指标: {metrics_path}")
    return metrics


# ────────────────────── 主函数 ──────────────────────

def main():
    parser = argparse.ArgumentParser(description="页面级检索 + 策略聚合评估")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eval-dataset-path", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True,
                        help="image_page_metadata.json 路径")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="单页图片目录 (image_page_path)")
    parser.add_argument("--results-dir", type=str, default="./eval_results")
    parser.add_argument("--pos-target-column", type=str, default="pos_target_for_deepseek")
    parser.add_argument("--score-agg", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--model-type", type=str, default="colpali",
                        choices=["colpali", "colqwen"],
                        help="模型类型：colpali 或 colqwen")
    parser.add_argument("--strategy", type=str, nargs="+", default=["all"],
                        help="策略（可多选）: first4 first2_last2 random4 uniform4 last4 all_pages, 或 all 表示全部")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--full-pool", action="store_true",
                        help="编码 metadata 中全部页面并在全库上检索；默认只编码正例文档页面")
    parser.add_argument("--pool-size", type=int, default=None, metavar="N",
                        help="检索池文档数。小于正例文档数则用正例数，大于全量则用全量，否则用 N（正例+随机负例文档）。不传则按 --full-pool 或正例池")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    if "all" in args.strategy:
        strategies = ALL_STRATEGIES
    else:
        strategies = args.strategy

    print("=" * 60)
    print("页面级检索 + 策略聚合评估")
    print(f"模型: {args.model_path} ({args.model_type})")
    print(f"策略: {', '.join(strategies)}")
    print(f"聚合: {args.score_agg}")
    print(f"正例字段: {args.pos_target_column}")
    if args.pool_size is not None:
        print(f"检索池文档数: --pool-size {args.pool_size}")
    print("=" * 60)

    doc_to_ordered_pages = load_doc_to_ordered_pages(args.metadata_path)
    eval_data = load_json(args.eval_dataset_path)
    doc_to_page_files = load_doc_to_page_files(args.metadata_path)
    all_docs = set(os.path.basename(doc_key).rsplit(".", 1)[0] for doc_key in doc_to_page_files)
    positive_docs = set()
    for item in eval_data:
        targets = item.get(args.pos_target_column, [])
        if isinstance(targets, str):
            targets = [targets]
        for t in targets:
            positive_docs.add(os.path.basename(t).rsplit(".", 1)[0])
    positive_docs &= all_docs

    pool_docs_for_encode: Optional[Set[str]] = None
    if args.pool_size is not None:
        num_pos, num_all = len(positive_docs), len(all_docs)
        effective = min(max(args.pool_size, num_pos), num_all)
        if effective <= num_pos:
            pool_docs_for_encode = positive_docs
            print(f"部分池: pool_size={args.pool_size} ≤ 正例文档数 {num_pos}，使用正例池 {len(pool_docs_for_encode)} 个文档")
        elif effective >= num_all:
            pool_docs_for_encode = all_docs
            print(f"部分池: pool_size={args.pool_size} ≥ 全量 {num_all}，使用全库 {len(pool_docs_for_encode)} 个文档")
        else:
            need_extra = effective - num_pos
            extra = set(random.sample(all_docs - positive_docs, min(need_extra, len(all_docs - positive_docs))))
            pool_docs_for_encode = positive_docs | extra
            print(f"部分池: 目标 {args.pool_size}，有效 {effective}（正例 {num_pos} + 负例 {len(extra)}），共 {len(pool_docs_for_encode)} 个文档")

    # 加载模型（只加载一次）
    model_service = ModelService(args.model_path, args.model_type)

    # 编码所有页面（只编码一次，所有策略复用）
    print("\n步骤 1: 编码所有页面...")
    page_embeddings = collect_and_encode_pages(
        args.eval_dataset_path, args.image_dir,
        args.metadata_path, args.pos_target_column,
        model_service,
        full_pool=args.full_pool,
        pool_docs=pool_docs_for_encode,
    )

    # 确定搜索池文档：pool_size 已体现在 pool_docs_for_encode，否则按 full_pool / 正例
    if pool_docs_for_encode is not None:
        doc_to_ordered_pages = {d: ps for d, ps in doc_to_ordered_pages.items() if d in pool_docs_for_encode}
        print(f"搜索池（部分池）: {len(doc_to_ordered_pages)} 个文档, "
              f"总页数: {sum(len(ps) for ps in doc_to_ordered_pages.values())}")
    elif args.full_pool:
        print(f"搜索池（全库）: {len(doc_to_ordered_pages)} 个文档, "
              f"总页数: {sum(len(ps) for ps in doc_to_ordered_pages.values())}")
    else:
        doc_to_ordered_pages = {d: ps for d, ps in doc_to_ordered_pages.items() if d in positive_docs}
        print(f"搜索池（正例文档）: {len(doc_to_ordered_pages)} 个文档, "
              f"总页数: {sum(len(ps) for ps in doc_to_ordered_pages.values())}")

    # 对每个策略执行评估
    print("\n步骤 2: 按策略评估...")
    all_metrics: Dict[str, Dict] = {}
    for strategy in strategies:
        print(f"\n{'─' * 40}")
        print(f"策略: {strategy}")
        print(f"{'─' * 40}")
        metrics = run_evaluation(
            model_service, page_embeddings, doc_to_ordered_pages,
            eval_data, strategy, args.score_agg,
            args.pos_target_column, args.results_dir, args.run_tag,
        )
        all_metrics[strategy] = metrics

    # 多策略对比汇总
    if len(strategies) > 1:
        print(f"\n{'=' * 60}")
        print("所有策略对比:")
        print(f"{'=' * 60}")
        header = f"{'策略':<16}{'P@1':>8}{'R@5':>8}{'MRR@10':>8}{'NDCG@5':>8}"
        print(header)
        print("-" * len(header))
        for s, m in all_metrics.items():
            print(f"{s:<16}{m.get('P@1', 0):>8.4f}{m.get('R@5', 0):>8.4f}"
                  f"{m.get('MRR@10', 0):>8.4f}{m.get('NDCG@5', 0):>8.4f}")

        tag = f"_{args.run_tag}" if args.run_tag else ""
        summary_path = os.path.join(args.results_dir, f"page_strategy_summary{tag}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)
        print(f"\n汇总保存: {summary_path}")


if __name__ == "__main__":
    main()
