"""
PDFA 多类别训练脚本：从「JSON + 按类别图片目录」加载，不修改原 colpali_engine 代码。
数据约定：
  - dataset_root/{category}/query_list_text_{category}_with_pos_target_for_deepseek_cleaned.json
  - dataset_root/{category}/pos_target_for_deepseek_images_first4/{stem}.png
  - JSON 中 pos_target_for_deepseek 为 ["0051569.txt", ...]，图片名为 0051569.png
"""
import argparse
import os
import json
import shutil
from pathlib import Path

import random
import sys
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import TrainingArguments, TrainerCallback
from PIL import Image
from typing import Any, Dict, List, Union

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
    ColbertPairwiseCELoss,
    CombinedMultiPositiveLoss,
    MultiPositiveInfoNCELoss,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.gpu_stats import print_summary
from colpali_engine.utils.local_model_merge import model_merge
from colpali_engine.utils.logger_config import setup_logger
from colpali_engine.collators.visual_retriever_collator import (
    VisualRetrieverCollator,
    MultiPositiveVisualRetrieverCollator,
)

logger = setup_logger("ColPali")


def is_main_process() -> bool:
    """分布式下仅 rank 0 为 True，单卡或未初始化时为 True。"""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# 用于「负例 = 非己正例」的 dataset/collator：同一 doc 只出现一次，避免被误当负例
POS_TARGET_IDS_KEY = "pos_target_ids"


class ColPaliEngineDatasetWithIds(ColPaliEngineDataset):
    """在返回的 sample 中增加 pos_target_ids（原始 doc_id 列表），供按 id 去重的 collator 使用。"""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = super().__getitem__(idx)
        raw = self.data[idx]
        ids = raw.get(self.pos_target_column_name)
        if ids is None:
            result[POS_TARGET_IDS_KEY] = []
        else:
            result[POS_TARGET_IDS_KEY] = ids if isinstance(ids, list) else [ids]
        return result


class DedupByDocIdCollator(MultiPositiveVisualRetrieverCollator):
    """
    多正例 collator：按 doc_id 去重，同一文档在 batch 中只出现一次。
    positive_mask[i,j]=True 当且仅当文档 j 是 query i 的正例，因此负例 = 真正「不是自己正例」的文档。
    """

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries: List[Union[str, Image.Image]] = []
        # (doc_id -> image)，同一 id 只保留一张图
        id_to_doc: Dict[str, Any] = {}
        # 每个 query 的正例 id 列表（用于构建 positive_mask）
        query_pos_ids: List[List[str]] = []

        for ex in examples:
            query = ex.get(ColPaliEngineDataset.QUERY_KEY)
            queries.append(random.choice(query) if isinstance(query, list) else query)
            pos_targets = ex.get(ColPaliEngineDataset.POS_TARGET_KEY)
            pos_ids = ex.get(POS_TARGET_IDS_KEY)
            if pos_targets is None:
                pos_targets = []
            if not isinstance(pos_targets, list):
                pos_targets = [pos_targets]
            if pos_ids is None:
                pos_ids = [str(i) for i in range(len(pos_targets))]
            if not isinstance(pos_ids, list):
                pos_ids = [pos_ids]
            selected = self._select_positives_by_id(pos_targets, pos_ids)
            selected_ids = selected["ids"]
            selected_docs = selected["docs"]
            for doc_id, doc in zip(selected_ids, selected_docs):
                id_to_doc[doc_id] = doc
            query_pos_ids.append(selected_ids)

        unique_ids = list(dict.fromkeys(id_to_doc))
        all_docs = [id_to_doc[i] for i in unique_ids]
        id_to_idx = {i: j for j, i in enumerate(unique_ids)}
        B_query = len(queries)
        B_doc = len(all_docs)
        positive_mask = torch.zeros(B_query, B_doc, dtype=torch.bool)
        for i, ids in enumerate(query_pos_ids):
            for doc_id in ids:
                if doc_id in id_to_idx:
                    positive_mask[i, id_to_idx[doc_id]] = True

        if all(q is None for q in queries):
            batch_query = None
        else:
            batch_query = self.auto_collate(queries, prefix=self.query_prefix)
        batch_docs = self.auto_collate(all_docs, prefix=self.pos_doc_prefix)
        return {**batch_query, **batch_docs, "positive_mask": positive_mask}

    def _select_positives_by_id(self, pos_targets: List, pos_ids: List) -> Dict[str, List]:
        n = min(len(pos_targets), len(pos_ids))
        pos_targets = pos_targets[:n]
        pos_ids = pos_ids[:n]
        if len(pos_targets) <= self.max_positives_per_query:
            return {"docs": pos_targets, "ids": pos_ids}
        if self.positive_sampling_strategy == "random":
            idx = random.sample(range(len(pos_targets)), self.max_positives_per_query)
        else:
            idx = list(range(self.max_positives_per_query))
        return {"docs": [pos_targets[i] for i in idx], "ids": [pos_ids[i] for i in idx]}


def _format_log_value(v) -> str:
    """浮点数：过小/过大用科学计数法，否则用小数，避免 learning_rate 等显示成 0.000001。"""
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.6e}"
        return f"{v:.6f}"
    return str(v)


class LossLoggingCallback(TrainerCallback):
    """每 logging_steps 将 loss 等打印到 stdout，便于 nohup 日志中看到。多卡时仅主进程打印，避免重复/交错。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not is_main_process():
            return
        parts = [f"{k}={_format_log_value(v)}" for k, v in sorted(logs.items())]
        print("\n  [train] " + "  ".join(parts), flush=True)
        sys.stdout.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        """用于诊断：确认已进入训练循环（已过 DataLoader/首 batch）。"""
        if is_main_process():
            print("\n  [train] Training loop started (on_train_begin).\n", flush=True)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# 从「类别/文件名」解析到图片路径并加载的 Corpus（不依赖 colpali_engine 的 Corpus）
# ---------------------------------------------------------------------------
class ImageDirCorpus:
    """
    doc_id 格式: "category/filename" 如 "biology/0051569.txt"
    图片路径: base_dir / category / image_subdir / {stem}.png
    """

    def __init__(
        self,
        base_dir: Path,
        image_subdir: str = "pos_target_for_deepseek_images_first4",
    ):
        self.base_dir = Path(base_dir)
        self.image_subdir = image_subdir

    def retrieve(self, doc_id: str) -> Image.Image:
        if "/" in doc_id:
            category, filename = doc_id.split("/", 1)
        else:
            category = ""
            filename = doc_id
        stem = Path(filename).stem
        path = self.base_dir / category / self.image_subdir / f"{stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path} (doc_id={doc_id})")
        return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# PDFA 多类别数据加载
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES = [
    "biology",
    "education",
    "finance",
    "government",
    "industrial",
    "legal",
    "research",
]

JSON_NAME_TEMPLATE = "query_list_text_{category}_with_pos_target_for_deepseek_cleaned.json"


def discover_categories(dataset_root: Path) -> list[str]:
    """发现 dataset_root 下存在约定 JSON 的子目录名。"""
    found = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        cat = p.name
        json_path = p / JSON_NAME_TEMPLATE.format(category=cat)
        if json_path.exists():
            found.append(cat)
    return found


def load_pdfa_train_set(
    dataset_root: str | Path,
    categories: list[str] | None = None,
    pos_target_column: str = "pos_target_for_deepseek",
    image_subdir: str = "pos_target_for_deepseek_images_first4",
    split_ratio: float = 0.0,
    seed: int = 42,
):
    """
    从 dataset_root 下多类别 JSON + 图片目录加载训练集（及可选验证集）。
    每条样本的 pos_target 为 ["category/xxx.txt", ...]，由 ImageDirCorpus 在 __getitem__ 时解析为 PIL Image。
    """
    dataset_root = Path(dataset_root)
    if categories is None:
        categories = discover_categories(dataset_root)
    if not categories:
        raise FileNotFoundError(f"No categories found under {dataset_root} (expected JSON: {JSON_NAME_TEMPLATE})")

    logger.info(f"📂 PDFA 数据根目录: {dataset_root}")
    logger.info(f"📂 类别: {categories}")

    corpus = ImageDirCorpus(dataset_root, image_subdir=image_subdir)
    datasets_per_cat = []

    for cat in categories:
        json_path = dataset_root / cat / JSON_NAME_TEMPLATE.format(category=cat)
        if not json_path.exists():
            logger.warning(f"跳过不存在的 JSON: {json_path}")
            continue
        ds = load_dataset("json", data_files={"train": str(json_path)}, split="train")
        # 列名可能为 pos_target_for_deepseek，统一为 pos_target 并加上 category 前缀
        if pos_target_column not in ds.column_names:
            raise ValueError(f"JSON 缺少列 {pos_target_column}: {json_path}")

        def map_pos(row):
            raw = row.get(pos_target_column) or []
            if not isinstance(raw, list):
                raw = [raw]
            # "xxx.txt" -> "category/xxx.txt"
            prefixed = [f"{cat}/{x}" for x in raw if x]
            return {"pos_target": prefixed}

        ds = ds.map(map_pos, remove_columns=[pos_target_column], num_proc=1, desc=f"[{cat}] map")
        # 过滤掉无正例的样本
        ds = ds.filter(lambda r: len(r["pos_target"]) > 0, num_proc=1, desc=f"[{cat}] filter")
        datasets_per_cat.append(ds)
        logger.info(f"  - {cat}: {len(ds)} 条（有正例）")

    if not datasets_per_cat:
        raise ValueError("没有加载到任何有效类别数据")

    full = concatenate_datasets(datasets_per_cat)
    logger.info(f"📊 合并后总样本数: {len(full)}")

    if split_ratio > 0.0:
        full = full.shuffle(seed=seed)
        n = len(full)
        eval_size = int(n * split_ratio)
        train_size = n - eval_size
        train_raw = full.select(range(train_size))
        eval_raw = full.select(range(train_size, n))
        logger.info(f"📊 训练/验证: {train_size} / {eval_size}")
        train_ds = ColPaliEngineDatasetWithIds(train_raw, corpus=corpus, pos_target_column_name="pos_target")
        eval_ds = ColPaliEngineDatasetWithIds(eval_raw, corpus=corpus, pos_target_column_name="pos_target")
        return train_ds, eval_ds
    else:
        train_ds = ColPaliEngineDatasetWithIds(full, corpus=corpus, pos_target_column_name="pos_target")
        return train_ds


def parse_args():
    p = argparse.ArgumentParser(description="PDFA 多类别 ColPali 训练（JSON + 图片目录）")
    p.add_argument("--dataset-root", type=str, required=True,
                   help="PDFA 数据根目录，下含 {category}/query_list_text_*_cleaned.json 与 {category}/pos_target_for_deepseek_images_first4/")
    p.add_argument("--categories", type=str, nargs="+", default=None,
                   help="类别列表，默认自动发现；例: --categories biology education legal")
    p.add_argument("--pos-target-column", type=str, default="pos_target_for_deepseek")
    p.add_argument("--image-subdir", type=str, default="pos_target_for_deepseek_images_first4")

    p.add_argument("--output-dir", type=str, default="/data/docpc_project/models/colpali_pdfa_finetuned")
    p.add_argument("--pretrained-model-name-or-path", type=str, default="/data/docpc_project/models/colpali_v1.3_merged")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"])
    p.add_argument("--loss", type=str, default="combined",
                   choices=["ce", "pairwise", "multi_positive", "combined"])
    p.add_argument("--use-multi-positive", action="store_true")
    p.add_argument("--max-positives", type=int, default=10)
    p.add_argument("--infonce-weight", type=float, default=1.0)
    p.add_argument("--listwise-weight", type=float, default=0.1)
    p.add_argument("--listwise-freq", type=int, default=5)

    p.add_argument("--peft", action="store_true")
    p.add_argument("--peft-r", type=int, default=32)
    p.add_argument("--peft-lora-alpha", type=int, default=32)
    p.add_argument("--peft-lora-dropout", type=float, default=0.1)
    p.add_argument("--peft-init-lora-weights", type=str, default="gaussian")
    p.add_argument("--peft-bias", type=str, default="none")
    p.add_argument("--peft-task-type", type=str, default="FEATURE_EXTRACTION")
    p.add_argument("--peft-target-modules", type=str,
                   default="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)")

    p.add_argument("--num-train-epochs", type=int, default=5)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=400, help="需为 eval-steps 的整数倍（若启用 load_best_model_at_end）")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--split-ratio", type=float, default=0.02, help="验证集比例，0 表示不切分")
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--dataloader-num-workers", type=int, default=8,
                   help="DataLoader worker 数；多卡时建议 0 避免卡住，单卡可设 4～8")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 多卡（torchrun）时按 LOCAL_RANK 绑定当前进程的 GPU，避免 barrier 时 NCCL 用错设备导致 hang
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(normalize_scores=False)
    elif args.loss == "multi_positive":
        loss_func = MultiPositiveInfoNCELoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
        )
    elif args.loss == "combined":
        loss_func = CombinedMultiPositiveLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            infonce_weight=args.infonce_weight,
            listwise_weight=args.listwise_weight,
            listwise_freq=args.listwise_freq,
            k=10,
            listwise_loss_type="approx_ndcg",
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    processor = ColQwen2_5_Processor.from_pretrained(
        args.pretrained_model_name_or_path,
        max_num_visual_tokens=768,
    )

    use_multi = args.use_multi_positive or args.loss in ["multi_positive", "combined"]
    if use_multi:
        collator = DedupByDocIdCollator(
            processor=processor,
            max_length=256,
            max_positives_per_query=args.max_positives,
            positive_sampling_strategy="random",
        )
    else:
        collator = VisualRetrieverCollator(processor=processor, max_length=256)

    dataset_result = load_pdfa_train_set(
        dataset_root=args.dataset_root,
        categories=args.categories,
        pos_target_column=args.pos_target_column,
        image_subdir=args.image_subdir,
        split_ratio=args.split_ratio,
    )

    if args.split_ratio > 0.0:
        train_dataset, eval_dataset = dataset_result
        run_eval = True
        eval_strategy = "steps"
    else:
        train_dataset = dataset_result
        eval_dataset = None
        run_eval = False
        eval_strategy = "no"

    model = ColQwen2_5.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.config.use_cache = False

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run_eval=run_eval,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=args.per_device_train_batch_size,
            eval_strategy=eval_strategy,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_drop_last=True,  # 多卡一致性（与 train_model.py 一致）
            max_grad_norm=args.max_grad_norm,
            save_steps=args.save_steps,
            logging_steps=5,
            logging_dir=os.path.join(args.output_dir, "logs"),
            report_to="none",
            eval_steps=args.eval_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.lr,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=run_eval,
            metric_for_best_model="eval_loss" if run_eval else None,
            greater_is_better=False if run_eval else None,
            # 多卡稳定（与 docpc_project train_model.py 一致）
            bf16=True,
            ddp_find_unused_parameters=False,
            ddp_bucket_cap_mb=200,
        ),
        peft_config=LoraConfig(
            r=args.peft_r,
            lora_alpha=args.peft_lora_alpha,
            lora_dropout=args.peft_lora_dropout,
            init_lora_weights=args.peft_init_lora_weights,
            bias=args.peft_bias,
            task_type=args.peft_task_type,
            target_modules=args.peft_target_modules,
        ) if args.peft else None,
    )
    config._collator = collator

    # 抽样检查
    sample = train_dataset[0]
    n_pos = len(sample.get("pos_target", []))
    logger.info(f"🔍 样本检查: query 存在={('query' in sample)}; 正例数={n_pos}")

    # 与 train_model.py 一致：所有 rank 都做 mkdir，仅 rank0 写文件，再 barrier 后一起进 train
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    if dist.is_initialized():
        dist.barrier()

    if args.trainer == "hf":
        # 使用子类注入 LossLoggingCallback，使 nohup 日志中能看到 loss
        class _ColModelTrainingWithLossLog(ColModelTraining):
            def train(self) -> None:
                trainer = ContrastiveTrainer(
                    model=self.model,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    args=self.config.tr_args,
                    data_collator=self.collator,
                    loss_func=self.config.loss_func,
                    is_vision_model=self.config.processor is not None,
                    callbacks=[LossLoggingCallback()],
                )
                trainer.args.remove_unused_columns = False
                if is_main_process():
                    print("\n  [train] About to call trainer.train() ...\n", flush=True)
                # 多卡时先同步再进 train()，避免某 rank 先碰到内部 barrier 而其他 rank 还在前面做 I/O 导致死等
                if dist.is_initialized():
                    dist.barrier()
                result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
                print_summary(result)

        trainer = _ColModelTrainingWithLossLog(config)
    else:
        trainer = ColModelTorchTraining(config)
    trainer.collator = config._collator
    trainer.train()
    trainer.save()

    if is_main_process():
        merge_model_path = os.path.join(args.output_dir, "sft-colqwen-merged")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_merge(args.output_dir, merge_model_path, device)
        logger.info(f"✅ 合并模型已保存: {merge_model_path}")
