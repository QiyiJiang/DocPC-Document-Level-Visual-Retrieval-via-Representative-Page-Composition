import random
from typing import Any, Dict, List, Union

from PIL.Image import Image
import torch

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    # Prefixes
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, (ColPaliProcessor,)):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        # Force padding to be on the right for ColPaliProcessor.
        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries: List[Union[None, str, Image]] = []
        pos_targets: List[Union[str, Image]] = []
        neg_targets: List[Union[str, Image]] = []

        # Parse the examples.
        for example in examples:
            assert ColPaliEngineDataset.QUERY_KEY in example, f"Missing {ColPaliEngineDataset.QUERY_KEY} in example."
            query = example[ColPaliEngineDataset.QUERY_KEY]
            sampled_query = random.choice(query) if isinstance(query, list) else query
            queries.append(sampled_query)

            assert ColPaliEngineDataset.POS_TARGET_KEY in example, (
                f"Missing {ColPaliEngineDataset.POS_TARGET_KEY} in example."
            )
            pos_tgt = example[ColPaliEngineDataset.POS_TARGET_KEY]
            sample_pos = random.choice(pos_tgt) if isinstance(pos_tgt, list) else pos_tgt
            pos_targets.append(sample_pos)

            neg_tgt = example.get(ColPaliEngineDataset.NEG_TARGET_KEY, None)
            if neg_tgt is not None:
                sampled_neg = random.choice(neg_tgt) if isinstance(neg_tgt, list) else neg_tgt
                neg_targets.append(sampled_neg)

        # Process queries.
        if all(q is None for q in queries):
            batch_query = None
        elif any(q is None for q in queries):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.auto_collate(queries, prefix=self.query_prefix)

        # Process targets.
        batch_pos_target = self.auto_collate(pos_targets, prefix=self.pos_doc_prefix)
        batch_neg_target = self.auto_collate(neg_targets, prefix=self.neg_doc_prefix) if neg_targets else {}

        return {
            **batch_query,
            **batch_pos_target,
            **batch_neg_target,
        }

    def auto_collate(self, batch: List[Union[str, Image]], prefix: str = "") -> Dict[str, Any]:
        """Automatically collate a batch of documents."""
        # Convert Document objects to their underlying data.
        if isinstance(batch[0], str):
            return self.collate_texts(batch, prefix=prefix)
        elif isinstance(batch[0], Image):
            return self.collate_images(batch, prefix=prefix)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}. Expected str or Image.")

    def collate_images(self, images: List[Image], prefix: str = "") -> Dict[str, Any]:
        """Collate images into a batch."""
        # Process images.
        batch_im = self.processor.process_images(images=images)
        # Prefix keys to avoid collisions.
        return prefix_keys(batch_im, prefix)

    def collate_texts(self, texts: List[str], prefix: str = "") -> Dict[str, Any]:
        """Collate texts into a batch."""
        # Process texts.
        batch_text = self.processor.process_queries(
            queries=texts,
            max_length=self.max_length,
        )
        # Prefix keys to avoid collisions.
        return prefix_keys(batch_text, prefix)


class MultiPositiveVisualRetrieverCollator(VisualRetrieverCollator):
    """
    Collator for training vision retrieval models with multi-positive training.
    """
    
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        max_positives_per_query: int = 5,
        positive_sampling_strategy: str = "random",
    ):
        super().__init__(processor, max_length)
        self.max_positives_per_query = max_positives_per_query
        self.positive_sampling_strategy = positive_sampling_strategy

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理多正例数据，创建正例掩码
        
        输入：
        - examples: 每个example包含query和多个pos_targets
        
        输出：
        - batch数据 + positive_mask张量
        """
        queries: List[Union[None, str, Image]] = []
        all_docs: List[Union[str, Image]] = []
        positive_mask_list: List[List[bool]] = []
        
        current_doc_idx = 0

        # 第一轮：收集所有查询和对应的正例
        for example in examples:
            assert ColPaliEngineDataset.QUERY_KEY in example, f"Missing {ColPaliEngineDataset.QUERY_KEY} in example."
            query = example[ColPaliEngineDataset.QUERY_KEY]
            sampled_query = random.choice(query) if isinstance(query, list) else query
            queries.append(sampled_query)

            assert ColPaliEngineDataset.POS_TARGET_KEY in example, (
                f"Missing {ColPaliEngineDataset.POS_TARGET_KEY} in example."
            )
            pos_targets = example[ColPaliEngineDataset.POS_TARGET_KEY]
            
            # 处理多个正例
            if isinstance(pos_targets, list):
                selected_positives = self._select_positives(pos_targets)
            else:
                selected_positives = [pos_targets]
            
            # 记录当前查询的正例掩码
            query_mask = []
            for doc in selected_positives:
                all_docs.append(doc)
                query_mask.append(True)  # 标记为正例
                
            positive_mask_list.append(query_mask)

        # 第二轮：为每个查询添加in-batch负例
        batch_size = len(queries)
        total_docs_per_query = len(all_docs)
        
        # 扩展掩码矩阵：每行代表一个查询，每列代表一个文档
        positive_mask_matrix = torch.zeros(batch_size, total_docs_per_query, dtype=torch.bool)
        
        doc_start_idx = 0
        for query_idx, query_mask in enumerate(positive_mask_list):
            doc_end_idx = doc_start_idx + len(query_mask)
            
            # 设置当前查询的正例
            positive_mask_matrix[query_idx, doc_start_idx:doc_end_idx] = torch.tensor(query_mask)
            
            doc_start_idx = doc_end_idx

        # 处理查询
        if all(q is None for q in queries):
            batch_query = None
        elif any(q is None for q in queries):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.auto_collate(queries, prefix=self.query_prefix)

        # 处理所有文档
        batch_docs = self.auto_collate(all_docs, prefix=self.pos_doc_prefix)

        return {
            **batch_query,
            **batch_docs,
            "positive_mask": positive_mask_matrix,  # [B_query, B_doc] 正例掩码
        }
    
    def _select_positives(self, pos_targets: List) -> List:
        """根据策略选择正例"""
        if self.positive_sampling_strategy == "all":
            return pos_targets[:self.max_positives_per_query]
        elif self.positive_sampling_strategy == "random":
            if len(pos_targets) <= self.max_positives_per_query:
                return pos_targets
            else:
                return random.sample(pos_targets, self.max_positives_per_query)
        else:
            raise ValueError(f"Unknown positive sampling strategy: {self.positive_sampling_strategy}")


class AdaptiveMultiPositiveCollator(MultiPositiveVisualRetrieverCollator):
    """
    Adaptive multi-positive collator, dynamically balance the positive-negative ratio.
    """
    
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        max_positives_per_query: int = 5,
        target_pos_neg_ratio: float = 0.3,  # 目标正负例比例
        add_hard_negatives: bool = True,     # 是否添加困难负例
        positive_sampling_strategy: str = "random",
    ):
        super().__init__(processor, max_length, max_positives_per_query, positive_sampling_strategy)
        self.target_pos_neg_ratio = target_pos_neg_ratio
        self.add_hard_negatives = add_hard_negatives

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced multi-positive collator, including in-batch hard negative mining.
        """
        base_batch = super().__call__(examples)
        
        if self.add_hard_negatives:
            # TODO: Add hard negative mining logic here.
            # For example, select hard negatives based on similarity, balance the positive-negative ratio, etc.
            pass
            
        return base_batch
