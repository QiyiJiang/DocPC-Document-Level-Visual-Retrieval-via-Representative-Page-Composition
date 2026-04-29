import os
import sys
import math
import json
import random
import argparse
import torch
import time
import ctypes
import numpy as np
from tqdm import tqdm
from PIL import Image
import concurrent.futures
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.models import ColPali, ColPaliProcessor
from typing import List, Dict, Optional

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType as WeaviateDataType
    from weaviate.classes.query import MetadataQuery, Filter
    from weaviate.config import AdditionalConfig
    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False

from pymilvus import MilvusClient, DataType as MilvusDataType

DEVICES = "cuda" if torch.cuda.is_available() else "cpu"
WEAVIATE_PORT = 8079
WEAVIATE_GRPC_PORT = 50051
WEAVIATE_TOP_K_RESULTS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_page_to_doc_mapping(metadata_path: str) -> Dict[str, str]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for entry in meta:
        page_name = entry.get("page_name", "")
        document_name = entry.get("document_name", "")
        if not page_name or not document_name:
            continue
        page_base = os.path.basename(page_name).rsplit(".", 1)[0]
        doc_base = os.path.basename(document_name).rsplit(".", 1)[0]
        out[page_base] = doc_base
    return out

def load_doc_to_pages_mapping(metadata_path: str) -> Dict[str, List[str]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for entry in meta:
        page_name = entry.get("page_name", "")
        document_name = entry.get("document_name", "")
        if not document_name:
            continue
        doc_key = os.path.basename(document_name)
        if doc_key not in out:
            out[doc_key] = []
        out[doc_key].append(os.path.basename(page_name))
    return out

def compute_metrics(
    retrieved: List[Dict],
    ground_truth: List[Dict],
    k_values: List[int],
    pos_target_column: str = "pos_target",
    metadata_path: Optional[str] = None,
) -> Dict:
    page_to_doc = load_page_to_doc_mapping(metadata_path) if metadata_path else None

    results_total = {k: {'precision': [], 'recall': [], 'mrr': [], 'ndcg': []} for k in k_values}

    for query_entry in retrieved:
        query = query_entry['query']
        predicted = query_entry.get('results', [])

        gt_entry = next((item for item in ground_truth if item['query'] == query), None)
        if not gt_entry:
            continue

        gt_items = gt_entry.get(pos_target_column, [])
        if isinstance(gt_items, str):
            gt_items = [gt_items]
        gt_names = set()
        for item in gt_items:
            base_name = os.path.basename(item).rsplit(".", 1)[0]
            gt_names.add(base_name)

        unique_predicted = []
        seen = set()
        for item in predicted:
            if item not in seen:
                unique_predicted.append(item)
                seen.add(item)

        def is_hit(pred_name: str) -> bool:
            pred_base = os.path.basename(str(pred_name)).rsplit(".", 1)[0]
            if page_to_doc is not None:
                doc_base = page_to_doc.get(pred_base, pred_base)
                return doc_base in gt_names
            return pred_base in gt_names

        def pred_to_doc_base(pred_name: str) -> str:
            pred_base = os.path.basename(str(pred_name)).rsplit(".", 1)[0]
            if page_to_doc is not None:
                return page_to_doc.get(pred_base, pred_base)
            return pred_base

        for k in k_values:
            top_k = unique_predicted[:k]

            if page_to_doc is not None:
                unique_gt_docs_hit = set()
                for name in top_k:
                    doc_base = pred_to_doc_base(name)
                    if doc_base in gt_names:
                        unique_gt_docs_hit.add(doc_base)
                num_hits = len(unique_gt_docs_hit)
                seen_gt_docs = set()
                dcg = 0.0
                for rank, name in enumerate(top_k):
                    doc_base = pred_to_doc_base(name)
                    if doc_base in gt_names and doc_base not in seen_gt_docs:
                        seen_gt_docs.add(doc_base)
                        dcg += 1.0 / math.log2(rank + 2)
            else:
                hits = [i for i, name in enumerate(top_k) if is_hit(name)]
                num_hits = len(hits)
                dcg = sum([1 / math.log2(rank + 2) for rank in hits])

            precision = num_hits / k if k > 0 else 0
            recall = num_hits / len(gt_names) if gt_names else 0

            mrr = 0
            for rank, name in enumerate(top_k):
                if is_hit(name):
                    mrr = 1 / (rank + 1)
                    break

            idcg = sum([1 / math.log2(i + 2) for i in range(min(len(gt_names), k))])
            ndcg = dcg / idcg if idcg > 0 else 0

            results_total[k]['precision'].append(precision)
            results_total[k]['recall'].append(recall)
            results_total[k]['mrr'].append(mrr)
            results_total[k]['ndcg'].append(ndcg)

    avg_results_total = {}
    for k in k_values:
        if len(results_total[k]['precision']) > 0:
            avg_results_total[f'P@{k}'] = round(sum(results_total[k]['precision']) / len(results_total[k]['precision']), 4)
            avg_results_total[f'R@{k}'] = round(sum(results_total[k]['recall']) / len(results_total[k]['recall']), 4)
            avg_results_total[f'MRR@{k}'] = round(sum(results_total[k]['mrr']) / len(results_total[k]['mrr']), 4)
            avg_results_total[f'NDCG@{k}'] = round(sum(results_total[k]['ndcg']) / len(results_total[k]['ndcg']), 4)
        else:
            avg_results_total[f'P@{k}'] = 0.0
            avg_results_total[f'R@{k}'] = 0.0
            avg_results_total[f'MRR@{k}'] = 0.0
            avg_results_total[f'NDCG@{k}'] = 0.0

    return avg_results_total

class MilvusColbertRetriever:
    def __init__(self, uri, collection_name, dim=128):
        self.collection_name = collection_name
        self.dim = dim
        print(f'连接到Milvus: 集合 {self.collection_name}')
        self.client = MilvusClient(uri=uri)
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            print(f'删除已存在的集合 {self.collection_name}')
            self.client.drop_collection(collection_name=self.collection_name)
        
        print(f'创建集合 {self.collection_name}')
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="id", datatype=MilvusDataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="seq_id", datatype=MilvusDataType.INT16)
        schema.add_field(field_name="doc_id", datatype=MilvusDataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="doc", datatype=MilvusDataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="image_embeddings", datatype=MilvusDataType.FLOAT_VECTOR, dim=self.dim)
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        print(f'为集合 {self.collection_name} 创建索引')
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="image_embeddings_index"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="image_embeddings",
            index_name="image_embeddings_index",
            index_type="FLAT",
            metric_type="IP",
        )
        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    @staticmethod
    def _suppress_stderr():
        class _SuppressStderr:
            def __enter__(self):
                self._stderr_fd = sys.stderr.fileno()
                self._saved_fd = os.dup(self._stderr_fd)
                self._devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self._devnull, self._stderr_fd)
                return self
            def __exit__(self, *args):
                os.dup2(self._saved_fd, self._stderr_fd)
                os.close(self._saved_fd)
                os.close(self._devnull)
        return _SuppressStderr()

    def insert(self, data: List[Dict]):
        print(f'插入 {len(data)} 条数据到集合 {self.collection_name}')

        total_vectors = 0
        BATCH_SIZE = 5000
        buffer = []
        self._doc_vecs_cache: Dict[str, list] = {}

        def flush_buffer():
            nonlocal total_vectors
            if not buffer:
                return
            with self._suppress_stderr():
                self.client.insert(self.collection_name, buffer)
            total_vectors += len(buffer)
            buffer.clear()
            time.sleep(0.05)

        for item in tqdm(data, desc="插入数据", total=len(data)):
            image_embeddings = item.get("image_embeddings", [])
            image_path = item.get("image_path", "")

            if len(image_embeddings.shape) > 1:
                vecs = [image_embeddings[i] for i in range(image_embeddings.shape[0])]
                for i, vec in enumerate(vecs):
                    buffer.append({
                        "image_embeddings": vec.tolist(),
                        "seq_id": i,
                        "doc_id": image_path,
                        "doc": image_path,
                    })
                    if len(buffer) >= BATCH_SIZE:
                        flush_buffer()
                if image_path not in self._doc_vecs_cache:
                    self._doc_vecs_cache[image_path] = []
                self._doc_vecs_cache[image_path].append(
                    np.array(image_embeddings, dtype=np.float32)
                )
            else:
                buffer.append({
                    "image_embeddings": image_embeddings.tolist(),
                    "seq_id": 0,
                    "doc_id": image_path,
                    "doc": image_path,
                })
                if len(buffer) >= BATCH_SIZE:
                    flush_buffer()
                if image_path not in self._doc_vecs_cache:
                    self._doc_vecs_cache[image_path] = []
                self._doc_vecs_cache[image_path].append(
                    np.array(image_embeddings, dtype=np.float32).reshape(1, -1)
                )

        flush_buffer()

        for doc_id in self._doc_vecs_cache:
            self._doc_vecs_cache[doc_id] = np.vstack(self._doc_vecs_cache[doc_id])
        print(f"总共插入了 {total_vectors} 个向量 (已缓存 {len(self._doc_vecs_cache)} 个文档到内存)")

        print(f"创建索引到集合 {self.collection_name}")
        with self._suppress_stderr():
            self.create_index()
        print(f"创建索引完成")

    def preload_all_doc_vectors(self, doc_ids: Optional[List[str]] = None):
        if hasattr(self, '_doc_vecs_cache') and self._doc_vecs_cache:
            print(f"文档向量已在内存中 ({len(self._doc_vecs_cache)} 个文档)，跳过预加载")
            return

        if not doc_ids:
            print("未提供 doc_ids 且无缓存，无法预加载")
            return

        print(f"预加载 {len(doc_ids)} 个文档向量到内存...")
        self._doc_vecs_cache: Dict[str, np.ndarray] = {}
        total_vecs = 0

        with self._suppress_stderr():
            for i, doc_id in enumerate(doc_ids):
                rows = self.client.query(
                    collection_name=self.collection_name,
                    filter=f"doc_id == '{doc_id}'",
                    output_fields=["image_embeddings"],
                    limit=16384,
                )
                if rows:
                    self._doc_vecs_cache[doc_id] = np.vstack(
                        [np.array(r["image_embeddings"], dtype=np.float32) for r in rows]
                    )
                    total_vecs += len(rows)
                if (i + 1) % 100 == 0:
                    print(f"  已加载 {i + 1}/{len(doc_ids)} 个文档")

        print(f"预加载完成: {len(self._doc_vecs_cache)} 个文档, {total_vecs} 个向量")

    def search(
        self,
        data: List[List[float]],
        limit: int = WEAVIATE_TOP_K_RESULTS,
        return_properties: List[str] = ['image_path']
    ):
        query_matrix = np.array(data, dtype=np.float32)

        if hasattr(self, '_doc_vecs_cache') and self._doc_vecs_cache:
            scores = []
            for doc_id, doc_vecs in self._doc_vecs_cache.items():
                sim = query_matrix @ doc_vecs.T
                total_score = float(sim.max(axis=1).sum())
                scores.append((total_score, doc_id))
        else:
            raise RuntimeError("请先调用 preload_all_doc_vectors() 预加载文档向量")

        scores.sort(key=lambda x: x[0], reverse=True)
        search_results = [
            {"image_path": doc_id, "score": score}
            for score, doc_id in scores[:limit]
        ]
        return type('SearchResponse', (), {'objects': search_results})()

    def load_collection(self):
        self.client.load_collection(collection_name=self.collection_name)

    def release_collection(self):
        self.client.release_collection(collection_name=self.collection_name)

    def drop_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

    def close(self):
        self.client.close()

class WeaviateDataBase:
    def __init__(self, collection_name):
        if not HAS_WEAVIATE:
            raise RuntimeError(
                "Weaviate is not installed. Please use --use-milvus or install weaviate."
            )
        self.collection_name = collection_name
        self.client = weaviate.connect_to_local(
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            additional_config=AdditionalConfig(timeout=(60, 180))
        )
    def create_collection(self):
        if not self.client.collections.exists(name=self.collection_name):
            print(f'创建集合 {self.collection_name}')
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="image_path", data_type=WeaviateDataType.TEXT),
                ],
                vectorizer_config=[
                    Configure.NamedVectors.none(
                        name="image_embeddings",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            multi_vector=Configure.VectorIndex.MultiVector.multi_vector()
                        )
                    )
                ]
            )

    def insert(self, data: List[Dict]):
        collect = self.client.collections.get(name=self.collection_name)
        with collect.batch.dynamic() as batch:
            for item in data:
                properties = {
                    "image_path": item.get("image_path", ""),
                }
                batch.add_object(
                    properties=properties,
                    vector={"image_embeddings": item.get("image_embeddings", [])}
                )

    def search(
        self,
        data: List[List[float]],
        limit: int = WEAVIATE_TOP_K_RESULTS,
        return_properties: List[str] = ['image_path']
    ):
        collect = self.client.collections.get(name=self.collection_name)

        filter_list = []
        filters = Filter.all_of(filter_list) if filter_list else None

        response = collect.query.near_vector(
            near_vector=data,
            target_vector="image_embeddings",
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
            return_properties=return_properties
        )
        
        search_results = []
        for obj in response.objects:
            image_path = obj.properties.get("image_path", "") if hasattr(obj, 'properties') else ""
            search_results.append({
                "image_path": image_path
            })
        
        return type('SearchResponse', (), {
            'objects': search_results
        })()

    def close(self):
        self.client.close()

class ModelService:
    """统一的模型服务类，通过 model_type 选择 ColPali 或 ColQwen"""
    
    def __init__(self, model_path: str, model_type: str = "colpali", max_pixels: int = None) -> None:
        print("================================================")
        print(f"从 {model_path} 加载模型 (类型: {model_type})")
        if model_type == "colpali":
            self.model = ColPali.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=DEVICES,
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model_path, use_fast=True)
        elif model_type == "colqwen":
            self.model = ColQwen2_5.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=DEVICES,
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(model_path, use_fast=True)
        else:
            raise ValueError(f"不支持的 model_type: {model_type}，请选择 colpali 或 colqwen")
        if max_pixels is not None:
            ip = getattr(self.processor, 'image_processor', None) \
                or getattr(getattr(self.processor, 'processor', None), 'image_processor', None)
            if ip is not None:
                old = ip.max_pixels
                ip.max_pixels = max_pixels
                print(f"已覆盖 max_pixels: {old:,} -> {max_pixels:,}")
   
    def multi_vectorize_image(self, img: Image.Image, resize: tuple = None) -> torch.Tensor:
        try:
            if resize is not None:
                img = img.resize(resize, Image.LANCZOS)
            image_batch = self.processor.process_images([img]).to(self.model.device)
            with torch.no_grad():
                image_embedding = self.model(**image_batch)
            return image_embedding[0].cpu().float().numpy()
        except Exception as e:
            raise RuntimeError(f"图像向量化失败: {str(e)}")

    def multi_vectorize_text(self, query: str) -> torch.Tensor:
        try:
            query_batch = self.processor.process_queries([query]).to(self.model.device)
            with torch.no_grad():
                query_embedding = self.model(**query_batch)
            return query_embedding[0].cpu().float().numpy()
        except Exception as e:
            raise RuntimeError(f"文本向量化失败: {str(e)}")

def generate_dataset_from_folder(
    eval_dataset_path: str,
    image_dir: str,
    collection_name: str,
    model_path: str,
    model_type: str = "colpali",
    use_milvus: bool = False,
    pos_target_column: str = "pos_target",
    milvus_uri: Optional[str] = None,
    metadata_path: Optional[str] = None,
    full_pool: bool = False,
    pool_size: Optional[int] = None,
    resize: tuple = None,
    max_pixels: int = None,
):
    if use_milvus:
        if not milvus_uri:
            raise ValueError("use_milvus=True 时必须传入 milvus_uri")
        model_service = ModelService(model_path=model_path, model_type=model_type, max_pixels=max_pixels)
        test_image = Image.new('RGB', (224, 224))
        test_embedding = model_service.multi_vectorize_image(test_image)
        dim = test_embedding.shape[-1] if len(test_embedding.shape) > 1 else test_embedding.shape[0]

        vector_client = MilvusColbertRetriever(uri=milvus_uri, collection_name=collection_name, dim=dim)
        vector_client.create_collection()
    else:
        vector_client = WeaviateDataBase(collection_name=collection_name)
        vector_client.create_collection()
    
    model_service = ModelService(model_path=model_path, model_type=model_type, max_pixels=max_pixels)
    data_results = []

    # 全量图片列表（用于 pool_size 或 full_pool）
    all_image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    all_base = set(os.path.basename(p).rsplit(".", 1)[0] for p in all_image_files)

    image_files = []
    if pool_size is not None:
        # 部分池模式：pool_size 控制检索池大小
        # 先算正例 base（仅统计在 image_dir 中存在的）
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_to_pages = load_doc_to_pages_mapping(metadata_path) if metadata_path else None
        positive_paths = set()
        for item in data:
            names = item.get(pos_target_column, "")
            if isinstance(names, str):
                names = [names]
            for name in names:
                doc_key = os.path.basename(name)
                if doc_to_pages is not None:
                    for page_name in doc_to_pages.get(doc_key, []):
                        positive_paths.add(os.path.join(image_dir, page_name))
                    if not doc_to_pages.get(doc_key):
                        positive_paths.add(os.path.join(image_dir, doc_key))
                else:
                    positive_paths.add(os.path.join(image_dir, doc_key))
        positive_base = set(os.path.basename(p).rsplit(".", 1)[0] for p in positive_paths) & all_base
        num_pos = len(positive_base)
        num_all = len(all_base)
        effective = min(max(pool_size, num_pos), num_all)
        if effective <= num_pos:
            pool_base = positive_base
            print(f"部分池模式: pool_size={pool_size} ≤ 正例数 {num_pos}，使用正例池 {len(pool_base)} 个图片")
        elif effective >= num_all:
            pool_base = all_base
            print(f"部分池模式: pool_size={pool_size} ≥ 全量 {num_all}，使用全库 {len(pool_base)} 个图片")
        else:
            need_extra = effective - num_pos
            extra_base = set(random.sample(all_base - positive_base, min(need_extra, len(all_base - positive_base))))
            pool_base = positive_base | extra_base
            print(f"部分池模式: 目标 {pool_size}，有效 {effective}（正例 {num_pos} + 负例 {len(extra_base)}），共 {len(pool_base)} 个图片")
        image_files = [p for p in all_image_files if os.path.basename(p).rsplit(".", 1)[0] in pool_base]
    elif full_pool:
        image_files = all_image_files
        print(f"全库模式: 找到 {len(image_files)} 个图片")
    else:
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_to_pages = load_doc_to_pages_mapping(metadata_path) if metadata_path else None
        for item in data:
            names = item.get(pos_target_column, "")
            if isinstance(names, str):
                names = [names]
            for name in names:
                doc_key = os.path.basename(name)
                if doc_to_pages is not None:
                    for page_name in doc_to_pages.get(doc_key, []):
                        image_files.append(os.path.join(image_dir, page_name))
                    if not doc_to_pages.get(doc_key):
                        image_files.append(os.path.join(image_dir, doc_key))
                else:
                    image_files.append(os.path.join(image_dir, doc_key))
        image_files = list(set(image_files))
        print(f"正例池模式: 找到 {len(image_files)} 个唯一的图片路径")
    
    processed_images = {}
    
    for image_file in tqdm(image_files, desc="生成数据集"):
        base_name = os.path.basename(image_file).split(".")[0]
        
        if base_name in processed_images:
            continue
            
        try:
            image_embeddings = model_service.multi_vectorize_image(Image.open(image_file), resize=resize)
            processed_images[base_name] = image_embeddings
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")
            continue

    for base_name, image_embeddings in processed_images.items():
        data_results.append({
            "image_path": base_name,
            "image_embeddings": image_embeddings
        })

    print(f"成功处理 {len(data_results)} 个唯一图片")
    vector_client.insert(data_results)
    return len(data_results), vector_client

def search_dataset(
    collection_name: str,
    model_path: str,
    model_type: str,
    eval_dataset_path: str,
    results_path: str,
    use_milvus: bool = False,
    pos_target_column: str = "pos_target",
    milvus_uri: Optional[str] = None,
    metadata_path: Optional[str] = None,
    score_agg: str = "max",
    cached_vector_client=None,
):
    if cached_vector_client is not None:
        vector_client = cached_vector_client
    elif use_milvus:
        if not milvus_uri:
            raise ValueError("use_milvus=True 时必须传入 milvus_uri")
        model_service = ModelService(model_path=model_path, model_type=model_type)
        test_image = Image.new('RGB', (224, 224))
        test_embedding = model_service.multi_vectorize_image(test_image)
        dim = test_embedding.shape[-1] if len(test_embedding.shape) > 1 else test_embedding.shape[0]
        vector_client = MilvusColbertRetriever(uri=milvus_uri, collection_name=collection_name, dim=dim)
    else:
        vector_client = WeaviateDataBase(collection_name=collection_name)

    model_service = ModelService(model_path=model_path, model_type=model_type)
    page_to_doc = load_page_to_doc_mapping(metadata_path) if metadata_path else None
    search_limit = 300 if page_to_doc else WEAVIATE_TOP_K_RESULTS

    if use_milvus and hasattr(vector_client, 'preload_all_doc_vectors'):
        vector_client.preload_all_doc_vectors()

    with open(eval_dataset_path, "r") as f:
        data = json.load(f)

    for item in tqdm(data, desc="搜索数据集"):
        query = item.get("query", "")
        if not query:
            continue

        query_embeddings = model_service.multi_vectorize_text(query)
        if len(query_embeddings.shape) > 1:
            query_vectors = [query_embeddings[i].tolist() for i in range(query_embeddings.shape[0])]
        else:
            query_vectors = [query_embeddings.tolist()]

        search_results = vector_client.search(query_vectors, limit=search_limit)

        page_results: List[tuple] = []
        if hasattr(search_results, 'objects'):
            for obj in search_results.objects:
                if isinstance(obj, dict):
                    name = str(obj.get("image_path", ""))
                    score = float(obj.get("score", 0))
                    if name:
                        page_results.append((name, score))
                elif hasattr(obj, 'properties') and hasattr(obj.properties, 'get'):
                    name = str(obj.properties.get("image_path", ""))
                    if name:
                        page_results.append((name, 0.0))

        if page_to_doc is not None and page_results:
            doc_scores: Dict[str, List[float]] = {}
            for page_name, score in page_results:
                doc_base = page_to_doc.get(page_name, page_name)
                if doc_base not in doc_scores:
                    doc_scores[doc_base] = []
                doc_scores[doc_base].append(score)

            doc_agg: Dict[str, float] = {}
            for doc_name, scores_list in doc_scores.items():
                if score_agg == "avg":
                    doc_agg[doc_name] = sum(scores_list) / len(scores_list)
                else:  # max
                    doc_agg[doc_name] = max(scores_list)

            sorted_docs = sorted(doc_agg.items(), key=lambda x: x[1], reverse=True)
            item["results"] = [doc for doc, _ in sorted_docs[:WEAVIATE_TOP_K_RESULTS]]
        else:
            item["results"] = [name for name, _ in page_results[:WEAVIATE_TOP_K_RESULTS]]

        if pos_target_column in item:
            item.pop(pos_target_column)

    with open(results_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    vector_client.close()

def evaluate_model(
    model_path: str,
    model_type: str,
    image_dir: str,
    eval_dataset_path: str,
    results_dir: str = "./eval_results",
    use_milvus: bool = False,
    pos_target_column: str = "pos_target",
    metadata_path: Optional[str] = None,
    score_agg: str = "max",
    run_tag: str = "",
    full_pool: bool = False,
    pool_size: Optional[int] = None,
    resize: tuple = None,
    max_pixels: int = None,
):
    os.makedirs(results_dir, exist_ok=True)

    random_number = random.randint(1, 1000000)
    collection_name = f"colpali_eval_datasets_{random_number}"
    if run_tag:
        results_path = os.path.join(results_dir, f"eval_{run_tag}.json")
    else:
        results_path = os.path.join(results_dir, f"reference_test_datasets_{random_number}.json")

    milvus_uri = None
    _saved_stderr_fd = None
    if use_milvus:
        db_dir = "db"
        os.makedirs(db_dir, exist_ok=True)
        milvus_uri = os.path.join(db_dir, f"milvus_{random_number}.db")

    print("=" * 60)
    print("开始模型评估")
    print(f"模型路径: {model_path}")
    print(f"模型类型: {model_type}")
    print(f"图片目录: {image_dir}")
    print(f"评估数据集: {eval_dataset_path}")
    print(f"集合名称: {collection_name}")
    print(f"使用数据库: {'Milvus' if use_milvus else 'Weaviate'}")
    if use_milvus:
        print(f"Milvus URI: {milvus_uri}")
    print(f"正例字段: {pos_target_column}")
    if pool_size is not None:
        print(f"检索池: --pool-size {pool_size}（小于正例数用正例数，大于全量用全量，否则用该值）")
    if metadata_path:
        print(f"Metadata 映射: {metadata_path} (正例=document_name, 预测=page_name)")
        print(f"分数聚合方式: {score_agg}")
    if pool_size is None:
        print(f"索引范围: {'全库(image_dir 下全部图片)' if full_pool else '仅正例文档'}")
    if resize:
        print(f"图片 resize: {resize[0]}x{resize[1]}")
    if max_pixels:
        print(f"max_pixels: {max_pixels:,}")
    print("=" * 60)

    print("\n步骤1: 生成向量数据库...")
    num_images, cached_vc = generate_dataset_from_folder(
        eval_dataset_path=eval_dataset_path,
        image_dir=image_dir,
        collection_name=collection_name,
        model_path=model_path,
        model_type=model_type,
        use_milvus=use_milvus,
        pos_target_column=pos_target_column,
        milvus_uri=milvus_uri,
        metadata_path=metadata_path,
        full_pool=full_pool,
        pool_size=pool_size,
        resize=resize,
        max_pixels=max_pixels,
    )
    print(f"成功处理 {num_images} 张图片")

    if use_milvus:
        sys.stderr.flush()
        _saved_stderr_fd = os.dup(2)
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
    
    print("\n步骤2: 执行搜索...")
    search_dataset(
        collection_name=collection_name,
        model_path=model_path,
        model_type=model_type,
        eval_dataset_path=eval_dataset_path,
        results_path=results_path,
        use_milvus=use_milvus,
        pos_target_column=pos_target_column,
        milvus_uri=milvus_uri,
        metadata_path=metadata_path,
        score_agg=score_agg,
        cached_vector_client=cached_vc,
    )
    print(f"搜索结果保存到: {results_path}")
    
    print("\n步骤3: 计算评估指标...")
    k_values = [1, 3, 5, 10]
    
    retrieved_data = load_json(results_path)
    ground_truth_data = load_json(eval_dataset_path)
    
    metrics = compute_metrics(
        retrieved_data, ground_truth_data, k_values,
        pos_target_column=pos_target_column,
        metadata_path=metadata_path,
    )
    
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    model_name = os.path.basename(os.path.dirname(model_path)) or os.path.basename(model_path)
    if run_tag:
        metrics_path = os.path.join(results_dir, f"metrics_{run_tag}.json")
    else:
        metrics_path = os.path.join(results_dir, f"{model_name}_metrics_{random_number}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\n评估指标保存到: {metrics_path}")

    if _saved_stderr_fd is not None:
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stderr_fd)

    if milvus_uri and os.path.exists(milvus_uri):
        os.remove(milvus_uri)
        lock_file = f".{os.path.basename(milvus_uri)}.lock"
        lock_path = os.path.join(os.path.dirname(milvus_uri), lock_file)
        if os.path.exists(lock_path):
            os.remove(lock_path)
        print(f"已清理 Milvus 文件: {milvus_uri}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="统一模型评估脚本（支持 ColPali / ColQwen）")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="colpali",
                        choices=["colpali", "colqwen"],
                        help="模型类型：colpali 或 colqwen（默认 colpali）")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--eval-dataset-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--use-milvus", action="store_true", help="使用Milvus而不是Weaviate")
    parser.add_argument(
        "--pos-target-column",
        type=str,
        default="pos_target",
        help="作为正例的 JSON 字段名，如 pos_target 或 pos_target_for_deepseek",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="page_name<->document_name 映射 JSON",
    )
    parser.add_argument(
        "--score-agg",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="同一文档多页分数聚合方式，默认 max",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="附加到结果文件名的标签（如类别名 biology）",
    )
    parser.add_argument(
        "--full-pool",
        action="store_true",
        help="索引 image_dir 下全部图片（全库检索）；默认只索引评估集中正例文档",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        metavar="N",
        help="检索池大小（图片数）。小于正例数则用正例数，大于全量则用全量，否则用 N（正例+随机负例）。不传则按 --full-pool 或正例池",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        metavar="WxH",
        help="编码前将图片 resize 到指定分辨率，格式 WxH（如 128x166、256x331、512x662）",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="覆盖 processor 的 max_pixels（如 1400000 支持 1024x1325）",
    )
    args = parser.parse_args()

    resize_tuple = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize_tuple = (int(w), int(h))

    metrics = evaluate_model(
        model_path=args.model_path,
        model_type=args.model_type,
        image_dir=args.image_dir,
        eval_dataset_path=args.eval_dataset_path,
        results_dir=args.results_dir,
        use_milvus=args.use_milvus,
        pos_target_column=args.pos_target_column,
        metadata_path=args.metadata_path,
        score_agg=args.score_agg,
        run_tag=args.run_tag,
        full_pool=args.full_pool,
        pool_size=args.pool_size,
        resize=resize_tuple,
        max_pixels=args.max_pixels,
    )

if __name__ == "__main__":
    main()
