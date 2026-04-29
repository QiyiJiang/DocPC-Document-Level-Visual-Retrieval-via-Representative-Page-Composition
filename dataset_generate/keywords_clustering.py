"""
对 data_text_keywords.json 中的关键词做 embedding + FAISS KMeans 聚类，
结果保存为 keywords_clusters.json，可选画肘部图选 k。
"""
import os

# 不使用物理 0 号卡：在首次 import 任何 CUDA 库前设置；设置后脚本内 cuda:0 = 物理第 1 张可见卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# 说明：PyTorch 日志里的 cuda:0 指「当前进程可见的第 1 张卡」，即物理 6 号卡，未使用物理 0 号卡
_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _gpus:
    print(f"INFO: CUDA_VISIBLE_DEVICES={_gpus} → 日志中 cuda:0 对应物理卡 {_gpus.split(',')[0].strip()}")

from pathlib import Path

import json
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============ 全部参数（在此修改） ============
SCRIPT_DIR = Path(__file__).resolve().parent
PDFA_DIR = SCRIPT_DIR / "pdfa_test"   # pdfa 数据目录
KEYWORDS_PATH = PDFA_DIR / "biology" / "data_text_keywords_biology.json"   # 关键词 JSON
CLUSTERS_PATH = PDFA_DIR / "biology" / "keywords_clusters_biology.json"   # 聚类结果保存路径
ELBOW_PATH = PDFA_DIR / "biology" / "elbow_biology.png"   # 肘部图保存路径

MODEL_NAME = "/data/docpc_project/models/Qwen3-Embedding-8B"   # 本地模型路径（或 HF 模型名）
N_CLUSTERS = None   # 聚类数 k；填 None 则自动用 sqrt(关键词数)，并限制在 2~100
PLOT_ELBOW = True  # 是否先画肘部图再聚类
ELBOW_K_MIN = 2
ELBOW_K_MAX = 100
ELBOW_STEP = 2

BATCH_SIZE = 64       # embedding 批大小
KMEANS_N_ITER = 25    # KMeans 迭代次数
KMEANS_SEED = 42
USE_GPU = True        # FAISS 是否用 GPU，不可用时会自动退到 CPU
# =============================================


def load_keywords(keywords_path: Path) -> list[str]:
    """从 data_text_keywords.json 格式中收集所有不重复关键词。"""
    with open(keywords_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    keywords_list = []
    for item in data_list:
        if item is not None and "keywords" in item:
            keywords_list.extend(item["keywords"])
    return list(set(keywords_list))


def encode_keywords(
    model_name: str,
    keywords_list: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """用 SentenceTransformer 对关键词做 embedding。"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        keywords_list,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def faiss_kmeans_clustering(
    embeddings: np.ndarray,
    keywords_list: list[str],
    n_clusters: int,
    save_path: Path,
    n_iter: int = 25,
    seed: int = 42,
    use_gpu: bool = True,
) -> list[dict]:
    """FAISS KMeans 聚类，按 cluster_id 汇总关键词并保存 JSON。"""
    d = embeddings.shape[1]
    try:
        kmeans = faiss.Kmeans(
            d, n_clusters, niter=n_iter, verbose=True, seed=seed, gpu=use_gpu
        )
    except Exception:
        logger.warning("FAISS GPU 不可用，使用 CPU")
        kmeans = faiss.Kmeans(
            d, n_clusters, niter=n_iter, verbose=True, seed=seed, gpu=False
        )
    kmeans.train(embeddings)
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()

    clusters = [
        {
            "cluster_id": int(i),
            "keywords": [
                keywords_list[j]
                for j in range(len(keywords_list))
                if labels[j] == i
            ],
        }
        for i in range(n_clusters)
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    logger.info("聚类结果已保存到 %s", save_path)
    return clusters


def plot_elbow(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 30,
    step: int = 1,
    save_path: Path | None = None,
    n_iter: int = 10,
    seed: int = 42,
    use_gpu: bool = True,
) -> None:
    """肘部法：在不同 k 下训练 KMeans，画「簇内平方和」SSE 曲线（手算，随 k 增大而递减，便于找肘部）。"""
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = ELBOW_PATH
    ks = list(range(k_min, min(k_max + 1, embeddings.shape[0]), step))
    if not ks:
        logger.warning("k 范围为空，跳过肘部图")
        return
    sse_list = []
    d = embeddings.shape[1]
    for k in ks:
        logger.info("Evaluating k=%d", k)
        try:
            kmeans = faiss.Kmeans(
                d, k, niter=n_iter, verbose=False, seed=seed, gpu=use_gpu
            )
        except Exception:
            kmeans = faiss.Kmeans(
                d, k, niter=n_iter, verbose=False, seed=seed, gpu=False
            )
        kmeans.train(embeddings)
        _, labels = kmeans.index.search(embeddings, 1)
        labels = labels.flatten()
        # 手算簇内平方和 SSE（每个点到其质心的距离平方和），随 k 增大应递减
        centroids = kmeans.centroids.reshape(k, d).astype(np.float32)
        sse = 0.0
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                diff = embeddings[mask] - centroids[c]
                sse += np.sum((diff ** 2))
        sse_list.append(float(sse))

    plt.figure(figsize=(8, 5))
    plt.plot(ks, sse_list, marker="o")
    plt.xlabel("k")
    plt.ylabel("SSE (簇内平方和)")
    plt.title("Elbow Method (SSE 随 k 递减，拐点即肘部)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    logger.info("肘部图已保存到 %s", save_path)


def main():
    # 加载关键词
    keywords_list = load_keywords(KEYWORDS_PATH)
    logger.info("去重后关键词数: %d", len(keywords_list))
    if len(keywords_list) < 2:
        logger.error("关键词数量不足，无法聚类")
        return

    # Embedding
    embeddings = encode_keywords(MODEL_NAME, keywords_list, batch_size=BATCH_SIZE)

    # 可选：肘部图
    if PLOT_ELBOW:
        plot_elbow(
            embeddings,
            k_min=ELBOW_K_MIN,
            k_max=min(ELBOW_K_MAX, len(keywords_list) - 1),
            step=ELBOW_STEP,
            save_path=ELBOW_PATH,
            use_gpu=USE_GPU,
        )

    # 确定 k：肘部图往往无明显拐点，故默认用「关键词数开平方」启发式，或手动设 N_CLUSTERS
    if N_CLUSTERS is not None:
        n_clusters = N_CLUSTERS
    else:
        n_clusters = int(np.clip(np.sqrt(len(keywords_list)), 2, 100))
        logger.info("N_CLUSTERS 未指定，使用 sqrt(关键词数) 得 n_clusters=%d", n_clusters)
    n_clusters = min(n_clusters, len(keywords_list))

    # 聚类并保存
    faiss_kmeans_clustering(
        embeddings,
        keywords_list,
        n_clusters=n_clusters,
        save_path=CLUSTERS_PATH,
        n_iter=KMEANS_N_ITER,
        seed=KMEANS_SEED,
        use_gpu=USE_GPU,
    )


if __name__ == "__main__":
    main()
