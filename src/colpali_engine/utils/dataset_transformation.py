import os
from typing import List, Tuple, cast, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image

from colpali_engine.data.dataset import ColPaliEngineDataset, Corpus
from colpali_engine.utils.logger_config import setup_logger

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"
logger = setup_logger("ColPali")

def load_train_set(dataset_name: str, pos_target_column_name: str = None, split_ratio: float = 0.0) -> Union[ColPaliEngineDataset, Tuple[ColPaliEngineDataset, ColPaliEngineDataset]]:
    """
    加载训练集，可选择自动分割验证集
    
    Args:
        dataset_name: 数据集名称
        pos_target_column_name: 正例列名
        split_ratio: 验证集比例，0.0表示不分割，0.05表示5%作为验证集
    
    Returns:
        如果split_ratio > 0: 返回(train_dataset, eval_dataset)
        如果split_ratio = 0: 返回train_dataset
    """
    base_path = dataset_name

    logger.info("="*100)
    logger.info(f"Loading dataset from {base_path}")
    # dataset = load_dataset(base_path, split="train")
    # dataset = load_dataset(
    #     "json",
    #     data_files={"train": os.path.join(base_path, "tiny_train_datasets.json")},
    #     split="train",
    # )
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(base_path, "train-*.parquet")
        },
        split="train",
    )
    logger.info("="*100)
    
    # 自动检测数据格式
    if pos_target_column_name is None:
        # 检查是否包含多正例格式
        if "pos_target" in dataset.column_names:
            pos_target_column_name = "pos_target"
            print("🔄 检测到多正例数据格式 (pos_target)")
        elif "image" in dataset.column_names:
            pos_target_column_name = "image"
            print("🔄 检测到单正例数据格式 (image)")
        else:
            raise ValueError("数据集必须包含 'pos_target' 或 'image' 列")
    
    # 检查数据格式并打印统计信息
    if pos_target_column_name == "pos_target":
        # 多正例数据统计
        pos_counts = []
        for item in dataset:
            pos_targets = item["pos_target"]
            if isinstance(pos_targets, list):
                pos_counts.append(len(pos_targets))
            else:
                pos_counts.append(1)
        
        print(f"📊 多正例数据统计:")
        print(f"  - 总样本数: {len(dataset)}")
        print(f"  - 平均正例数: {sum(pos_counts)/len(pos_counts):.2f}")
        print(f"  - 正例数范围: {min(pos_counts)} - {max(pos_counts)}")
    else:
        print(f"📊 单正例数据统计:")
        print(f"  - 总样本数: {len(dataset)}")
    
    # 如果指定了切分比例，进行数据集分割
    if split_ratio > 0.0:
        logger.info(f"🔄 自动切分验证集，比例: {split_ratio:.1%}")
        
        # 计算切分点
        total_size = len(dataset)
        eval_size = int(total_size * split_ratio)
        train_size = total_size - eval_size
        
        logger.info(f"  - 训练集大小: {train_size}")
        logger.info(f"  - 验证集大小: {eval_size}")
        
        # 随机切分数据集
        dataset = dataset.shuffle(seed=42)
        train_dataset_raw = dataset.select(range(train_size))
        eval_dataset_raw = dataset.select(range(train_size, total_size))
        
        # 创建ColPaliEngineDataset
        train_dataset = ColPaliEngineDataset(train_dataset_raw, pos_target_column_name=pos_target_column_name)
        eval_dataset = ColPaliEngineDataset(eval_dataset_raw, pos_target_column_name=pos_target_column_name)
        
        return train_dataset, eval_dataset
    else:
        # 不切分，只返回训练集
        train_dataset = ColPaliEngineDataset(dataset, pos_target_column_name=pos_target_column_name)
        return train_dataset


def load_eval_set(dataset_path) -> ColPaliEngineDataset:
    dataset = load_dataset(dataset_path, split="test")

    return dataset


def load_train_set_ir(num_negs=0) -> ColPaliEngineDataset:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    corpus_data = load_dataset(base_path + "colpali-corpus", split="train")
    corpus = Corpus(corpus_data=corpus_data, doc_column_name="image")

    dataset = load_dataset(base_path + "colpali-queries", split="train")

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    if num_negs > 0:
        # keep only top 5 negative passages
        dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:num_negs]})
    print("Dataset size after filtering:", len(dataset))

    train_dataset = ColPaliEngineDataset(
        data=dataset,
        corpus=corpus,
        pos_target_column_name="positive_passages",
        neg_target_column_name="negative_passages" if num_negs else None,
    )

    return train_dataset


def load_train_set_detailed() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_train_set_with_tabfquad() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docmatix_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "docmatix-ir", split="train"))
    # dataset = dataset.select(range(100500))

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "Docmatix", "images", split="train"))

    return ds_dict, anchor_ds, "docmatix"


def load_wikiss() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "wiki-ss-nq", data_files="train.jsonl", split="train"))
    # dataset = dataset.select(range(400500))
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "wiki-ss-corpus", split="train"))

    return ds_dict, anchor_ds, "wikiss"


def load_train_set_with_docmatix() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
        "Docmatix_filtered_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot: List[Dataset] = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = concatenate_datasets(ds_tot)
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docvqa_dataset() -> DatasetDict:
    if USE_LOCAL_DATASET:
        dataset_doc = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="test"))
    else:
        dataset_doc = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test"))

    # concatenate the two datasets
    dataset = concatenate_datasets([dataset_doc, dataset_info])
    dataset_eval = concatenate_datasets([dataset_doc_eval, dataset_info_eval])
    # sample 100 from eval dataset
    dataset_eval = dataset_eval.shuffle(seed=42).select(range(200))

    # rename question as query
    dataset = dataset.rename_column("question", "query")
    dataset_eval = dataset_eval.rename_column("question", "query")

    # create new column image_filename that corresponds to ucsf_document_id if not None, else image_url
    dataset = dataset.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )
    dataset_eval = dataset_eval.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )

    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    return ds_dict


def load_dummy_dataset() -> List[DatasetDict]:
    # create a dataset from the queries and images
    queries_1 = ["What is the capital of France?", "What is the capital of Germany?"]
    queries_2 = ["What is the capital of Italy?", "What is the capital of Spain?"]

    images_1 = [Image.new("RGB", (100, 100)) for _ in range(2)]
    images_2 = [Image.new("RGB", (120, 120)) for _ in range(2)]

    dataset_1 = Dataset.from_list([{"query": q, "image": i} for q, i in zip(queries_1, images_1)])
    dataset_2 = Dataset.from_list([{"query": q, "image": i} for q, i in zip(queries_2, images_2)])

    return DatasetDict(
        {
            "train": DatasetDict({"dataset_1": dataset_1, "dataset_2": dataset_2}),
            "test": DatasetDict({"dataset_1": dataset_2, "dataset_2": dataset_1}),
        }
    )


def load_multi_qa_datasets() -> List[DatasetDict]:
    dataset_args = [
        ("vidore/colpali_train_set"),
        ("llamaindex/vdr-multilingual-train", "de"),
        ("llamaindex/vdr-multilingual-train", "en"),
        ("llamaindex/vdr-multilingual-train", "es"),
        ("llamaindex/vdr-multilingual-train", "fr"),
        ("llamaindex/vdr-multilingual-train", "it"),
    ]

    train_datasets = {}
    test_datasets = {}
    for args in dataset_args:
        dataset_name = args[0] + "_" + args[1]
        dataset = load_dataset(*args)
        if "test" in dataset:
            train_datasets[dataset_name] = dataset["train"]
            test_datasets[dataset_name] = dataset["test"]
        else:
            train_dataset, test_dataset = dataset.split_by_ratio(test_size=200)
            train_datasets[dataset_name] = train_dataset
            test_datasets[dataset_name] = test_dataset

    return DatasetDict({"train": DatasetDict(train_datasets), "test": DatasetDict(test_datasets)})


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test")
        return dataset


# def load_test_train_set(dataset_name: str) -> ColPaliEngineDataset:
#     """
#     加载测试用的多正例数据集
#     """
#     print("="*100)
#     print("🧪 加载测试数据集...")
    
#     # 尝试使用测试数据生成器创建数据
#     try:
#         from test_data_generator import create_in_memory_test_dataset
#         dataset = create_in_memory_test_dataset(num_samples=20)  # 创建20个测试样本
#         print(f"✅ 成功生成 {len(dataset)} 个测试样本")
        
#         # 显示数据统计
#         pos_counts = []
#         for item in dataset:
#             pos_targets = item["pos_target"]
#             if isinstance(pos_targets, list):
#                 pos_counts.append(len(pos_targets))
#             else:
#                 pos_counts.append(1)
        
#         print(f"📊 数据统计:")
#         print(f"  - 平均正例数: {sum(pos_counts)/len(pos_counts):.2f}")
#         print(f"  - 正例数范围: {min(pos_counts)} - {max(pos_counts)}")
        
#     except ImportError:
#         print("❌ 无法导入测试数据生成器，使用空数据集")
#         # 如果导入失败，创建一个简单的测试数据集
#         from PIL import Image
#         simple_data = []
#         for i in range(10):
#             # 创建简单的测试图像
#             img1 = Image.new('RGB', (224, 224), color=(255, 0, 0))
#             img2 = Image.new('RGB', (224, 224), color=(0, 255, 0))
#             simple_data.append({
#                 "query": f"测试查询 {i+1}",
#                 "pos_target": [img1, img2]  # 每个查询2个正例
#             })
        
#         from datasets import Dataset
#         dataset = Dataset.from_list(simple_data)
#         print(f"✅ 创建简单测试数据集，包含 {len(dataset)} 个样本")
    
#     print("="*100)
    
#     # 转换为ColPaliEngineDataset
#     train_dataset = ColPaliEngineDataset(dataset, pos_target_column_name="pos_target")
#     return train_dataset


# if __name__ == "__main__":
#     ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
#     print(ds)
