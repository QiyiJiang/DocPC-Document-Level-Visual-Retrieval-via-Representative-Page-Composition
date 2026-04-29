try:
    import weaviate
except ImportError:
    weaviate = None
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.config import AdditionalConfig
from typing import List, Dict, Optional
import os
from tqdm import tqdm
import json
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import random
import math

from colpali_engine.utils.logger_config import setup_logger
logger = setup_logger("ColPali")

DEVICES = "cuda" if torch.cuda.is_available() else "cpu"
WEAVIATE_PORT = 8079
WEAVIATE_GRPC_PORT = 50051
WEAVIATE_TOP_K_RESULTS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_json(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_metrics(retrieved: List[Dict], ground_truth: List[Dict], k_values: List[int]) -> Dict:
    """
    计算Precision@K, Recall@K, MRR@K, 和 NDCG@K
    适用于普通版本的数据格式，使用pos_target字段作为ground truth
    """
    results_total = {k: {'precision': [], 'recall': [], 'mrr': [], 'ndcg': []} for k in k_values}
    
    for query_entry in retrieved:
        query = query_entry['query']
        predicted = query_entry.get('results', [])

        # 找到对应的ground truth entry
        gt_entry = next((item for item in ground_truth if item['query'] == query), None)
        if not gt_entry:
            continue
        
        # 使用pos_target字段作为ground truth，并提取basename
        gt_items = gt_entry.get('pos_target', [])
        gt_names = set()
        for item in gt_items:
            # 取image的basename
            base_name = os.path.basename(item).split(".")[0]
            gt_names.add(base_name)
        
        # 去重预测结果
        unique_predicted = []
        seen = set()
        for item in predicted:
            if item not in seen:
                unique_predicted.append(item)
                seen.add(item)

        for k in k_values:
            top_k = unique_predicted[:k]

            # 命中索引
            hits = [i for i, name in enumerate(top_k) if name in gt_names]
            num_hits = len(hits)

            precision = num_hits / k if k > 0 else 0
            recall = num_hits / len(gt_names) if gt_names else 0

            # MRR@K
            mrr = 0
            for rank, name in enumerate(top_k):
                if name in gt_names:
                    mrr = 1 / (rank + 1)
                    break

            # NDCG@K
            dcg = sum([1 / math.log2(rank + 2) for rank in hits])
            idcg = sum([1 / math.log2(i + 2) for i in range(min(len(gt_names), k))])
            ndcg = dcg / idcg if idcg > 0 else 0

            results_total[k]['precision'].append(precision)
            results_total[k]['recall'].append(recall)
            results_total[k]['mrr'].append(mrr)
            results_total[k]['ndcg'].append(ndcg)

    # 平均结果
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

class WeaviateDataBase:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        logger.info(f'连接到Weaviate: {WEAVIATE_PORT} GRPC {WEAVIATE_GRPC_PORT} 集合 {self.collection_name}')
        self.client = weaviate.connect_to_local(port=WEAVIATE_PORT, 
                                                grpc_port=WEAVIATE_GRPC_PORT,
                                                additional_config=AdditionalConfig(timeout=(60, 180)))

    def create_collection(self):
        """创建集合，如果集合已存在则不做任何操作"""
        if not self.client.collections.exists(name=self.collection_name):
            logger.info(f'创建集合 {self.collection_name}')
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="image_path", data_type=DataType.TEXT),
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
        """插入向量数据"""
        collect = self.client.collections.get(name=self.collection_name)
        with collect.batch.dynamic() as batch:
            for item in data:
                # 构建属性字典
                properties = {
                    "image_path": item.get("image_path", ""),
                }
                # 添加对象到批处理
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

        # 动态构建过滤器列表
        filter_list = []

        # 组合过滤器
        filters = Filter.all_of(filter_list) if filter_list else None

        response = collect.query.near_vector(
            near_vector=data,
            target_vector="image_embeddings",
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
            return_properties=return_properties
        )
        
        return response

    def close(self):
        self.client.close()

class ColQwenService:
    """ColQwen服务类，用于处理图像和文本的多向量嵌入"""
    
    def __init__(self, model_path: str = "./base_model/colqwen2.5-v0.2") -> None:
        """初始化ColQwen模型和处理器"""
        logger.info("================================================")
        logger.info(f"从 {model_path} 加载模型")
        self.model = ColQwen2_5.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=DEVICES,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_path)

    def multi_vectorize_image(self, img: Image.Image) -> torch.Tensor:
        """将PIL图像转换为多向量表示"""
        try:
            image_batch = self.processor.process_images([img]).to(self.model.device)
            with torch.no_grad():
                image_embedding = self.model(**image_batch)
            return image_embedding[0].cpu().float().numpy()
        except Exception as e:
            raise RuntimeError(f"图像向量化失败: {str(e)}")

    def multi_vectorize_text(self, query: str) -> torch.Tensor:
        """将文本查询转换为多向量表示"""
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
    model_path: str
    ):
    """
    从文件夹直接生成数据集
    image_dir: 包含PNG图片的文件夹路径
    """
    weaviate_client = WeaviateDataBase(collection_name=collection_name)
    weaviate_client.create_collection()
    colqwen_service = ColQwenService(model_path=model_path)
    data_results = []

    # 获取eval_dataset_path文件夹中的所有PNG文件
    image_files = []
    with open(eval_dataset_path, "r") as f:
        data = json.load(f)
    for item in data:
        image_names = item.get("pos_target", "")
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            image_files.append(image_path)
    
    # 去重图片文件列表
    image_files = list(set(image_files))
    logger.info(f"找到 {len(image_files)} 个唯一的PNG文件")
    
    # 使用字典避免重复处理
    processed_images = {}
    
    for image_file in tqdm(image_files, desc="生成数据集"):
        # 取image的basename（不包含扩展名）
        base_name = os.path.basename(image_file).split(".")[0]
        
        # 如果已经处理过这个图片，跳过
        if base_name in processed_images:
            continue
            
        try:
            image_embeddings = colqwen_service.multi_vectorize_image(Image.open(image_file))
            processed_images[base_name] = image_embeddings
        except Exception as e:
            logger.info(f"处理图片 {image_file} 时出错: {e}")
            continue

    # 将处理结果转换为列表格式
    for base_name, image_embeddings in processed_images.items():
        data_results.append({
            "image_path": base_name,
            "image_embeddings": image_embeddings
        })

    logger.info(f"成功处理 {len(data_results)} 个唯一图片")
    weaviate_client.insert(data_results)
    weaviate_client.close()
    return len(data_results)

def search_dataset(collection_name: str,
                   model_path: str,
                   eval_dataset_path: str,
                   results_path: str):
    """搜索数据集并生成结果"""
    weaviate_client = WeaviateDataBase(collection_name=collection_name)
    colqwen_service = ColQwenService(model_path=model_path)
    
    with open(eval_dataset_path, "r") as f:
        data = json.load(f)
    
    for item in tqdm(data, desc="搜索数据集"):
        query = item.get("query", "")
        if not query:
            continue
            
        query_embeddings = colqwen_service.multi_vectorize_text(query)
        search_results = weaviate_client.search(query_embeddings)

        answer = [
            str(obj.properties["image_path"])
            for obj in search_results.objects
            if obj.properties.get("image_path") is not None
        ]
        item["results"] = answer
        item.pop("keyword_list")
        item.pop("pos_target")
    
    with open(results_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    weaviate_client.close()

def evaluate_model(model_path: str,
                   image_dir: str,
                   eval_dataset_path: str,
                   results_dir: str = "./eval_results"):
    """
    完整的模型评估流程
    
    Args:
        model_path: 模型路径
        image_dir: 包含PNG图片的文件夹路径
        eval_dataset_path: 评估数据集JSON文件路径
        results_dir: 结果保存目录
    """
    # 创建结果目录
    if weaviate is None:
        raise RuntimeError("weaviate is not installed, cannot run enhanced evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成随机集合名
    random_number = random.randint(1, 1000000)
    collection_name = f"colpali_eval_datasets_{random_number}"
    results_path = os.path.join(results_dir, f"reference_test_datasets_{random_number}.json")
    
    logger.info("=" * 60)
    logger.info("开始模型评估")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"图片目录: {image_dir}")
    logger.info(f"评估数据集: {eval_dataset_path}")
    logger.info(f"集合名称: {collection_name}")
    logger.info("=" * 60)
    
    # 步骤1: 从文件夹生成数据集
    logger.info("\n步骤1: 生成向量数据库...")
    num_images = generate_dataset_from_folder(
        eval_dataset_path=eval_dataset_path,
        image_dir=image_dir,
        collection_name=collection_name,
        model_path=model_path
    )
    logger.info(f"成功处理 {num_images} 张图片")
    
    # 步骤2: 搜索数据集
    logger.info("\n步骤2: 执行搜索...")
    search_dataset(
        collection_name=collection_name,
        model_path=model_path,
        eval_dataset_path=eval_dataset_path,
        results_path=results_path
    )
    logger.info(f"搜索结果保存到: {results_path}")
    
    # 步骤3: 计算评估指标
    logger.info("\n步骤3: 计算评估指标...")
    k_values = [1, 3, 5, 10]
    
    retrieved_data = load_json(results_path)
    ground_truth_data = load_json(eval_dataset_path)
    
    metrics = compute_metrics(retrieved_data, ground_truth_data, k_values)
    
    logger.info("\n" + "=" * 60)
    logger.info("评估结果:")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")
    
    # 保存评估结果
    metrics_path = os.path.join(results_dir, f"metrics_{random_number}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info(f"\n评估指标保存到: {metrics_path}")
    
    return metrics

def main():
    """主函数示例"""
    # 配置参数
    model_path = "/workspace/docpc/checkpoint/train_gold_tiny_0701/ft_colqwen-merged"
    image_dir = "/workspace/datasets/gold_datasets/image_path"  # 包含PNG图片的文件夹
    eval_dataset_path = "/workspace/datasets/gold_datasets/test_datasets.json"
    results_dir = "/workspace/docpc/eval_results"
    
    # 执行评估
    metrics = evaluate_model(
        model_path=model_path,
        image_dir=image_dir,
        eval_dataset_path=eval_dataset_path,
        results_dir=results_dir
    )

if __name__ == "__main__":
    main() 