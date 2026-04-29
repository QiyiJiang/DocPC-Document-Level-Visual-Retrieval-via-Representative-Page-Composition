"""
PDFA 多类别训练脚本（无正例去重版）。
与 train_pdfa_colpali.py 的区别：
  - 不使用 DedupByDocIdCollator
  - combined/multi_positive 使用 MultiPositiveVisualRetrieverCollator（支持多正例 + positive_mask），
    但不做正例去重：query_A 的正例文档仍然可以作为 query_B 的负例
  - ce/pairwise 使用 VisualRetrieverCollator（单正例）

数据约定同 train_pdfa_colpali.py：
  - dataset_root/{category}/query_list_text_{category}_with_pos_target_for_deepseek_cleaned.json
  - dataset_root/{category}/{image_subdir}/{stem}.png
"""
import argparse
import os
import json
import shutil
from pathlib import Path

import sys
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import TrainingArguments, TrainerCallback
from PIL import Image

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
    ColbertPairwiseCELoss,
    CombinedMultiPositiveLoss,
    MultiPositiveInfoNCELoss,
)
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

logger = setup_logger("ColPali-NoDedup")


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _format_log_value(v) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.6e}"
        return f"{v:.6f}"
    return str(v)


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not is_main_process():
            return
        parts = [f"{k}={_format_log_value(v)}" for k, v in sorted(logs.items())]
        print("\n  [train] " + "  ".join(parts), flush=True)
        sys.stdout.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        if is_main_process():
            print("\n  [train] Training loop started (on_train_begin).\n", flush=True)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
class ImageDirCorpus:
    """
    doc_id 格式: "category/filename" 如 "biology/0051569.txt"
    图片路径: base_dir / category / image_subdir / {stem}.png
    """

    def __init__(self, base_dir: Path, image_subdir: str = "pos_target_for_deepseek_images_first4"):
        self.base_dir = Path(base_dir)
        self.image_subdir = image_subdir

    def retrieve(self, doc_id: str) -> Image.Image:
        if "/" in doc_id:
            category, filename = doc_id.split("/", 1)
        else:
            category, filename = "", doc_id
        stem = Path(filename).stem
        path = self.base_dir / category / self.image_subdir / f"{stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path} (doc_id={doc_id})")
        return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES = ["biology", "education", "finance", "government", "industrial", "legal", "research"]
JSON_NAME_TEMPLATE = "query_list_text_{category}_with_pos_target_for_deepseek_cleaned.json"


def discover_categories(dataset_root: Path) -> list[str]:
    found = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        cat = p.name
        if (p / JSON_NAME_TEMPLATE.format(category=cat)).exists():
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
    dataset_root = Path(dataset_root)
    if categories is None:
        categories = discover_categories(dataset_root)
    if not categories:
        raise FileNotFoundError(f"No categories found under {dataset_root}")

    logger.info(f"PDFA root: {dataset_root}, categories: {categories}")
    corpus = ImageDirCorpus(dataset_root, image_subdir=image_subdir)
    datasets_per_cat = []

    for cat in categories:
        json_path = dataset_root / cat / JSON_NAME_TEMPLATE.format(category=cat)
        if not json_path.exists():
            logger.warning(f"skip missing: {json_path}")
            continue
        ds = load_dataset("json", data_files={"train": str(json_path)}, split="train")
        if pos_target_column not in ds.column_names:
            raise ValueError(f"JSON missing column {pos_target_column}: {json_path}")

        def map_pos(row):
            raw = row.get(pos_target_column) or []
            if not isinstance(raw, list):
                raw = [raw]
            return {"pos_target": [f"{cat}/{x}" for x in raw if x]}

        ds = ds.map(map_pos, remove_columns=[pos_target_column], num_proc=1, desc=f"[{cat}] map")
        ds = ds.filter(lambda r: len(r["pos_target"]) > 0, num_proc=1, desc=f"[{cat}] filter")
        datasets_per_cat.append(ds)
        logger.info(f"  {cat}: {len(ds)} samples")

    if not datasets_per_cat:
        raise ValueError("No valid category data loaded")

    full = concatenate_datasets(datasets_per_cat)
    logger.info(f"Total samples: {len(full)}")

    if split_ratio > 0.0:
        full = full.shuffle(seed=seed)
        n = len(full)
        eval_size = int(n * split_ratio)
        train_size = n - eval_size
        train_raw = full.select(range(train_size))
        eval_raw = full.select(range(train_size, n))
        logger.info(f"Train/Eval: {train_size} / {eval_size}")
        train_ds = ColPaliEngineDataset(train_raw, corpus=corpus, pos_target_column_name="pos_target")
        eval_ds = ColPaliEngineDataset(eval_raw, corpus=corpus, pos_target_column_name="pos_target")
        return train_ds, eval_ds
    else:
        train_ds = ColPaliEngineDataset(full, corpus=corpus, pos_target_column_name="pos_target")
        return train_ds


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PDFA 训练（无正例去重，标准 in-batch negatives）")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--pos-target-column", type=str, default="pos_target_for_deepseek")
    p.add_argument("--image-subdir", type=str, default="pos_target_for_deepseek_images_first4")

    p.add_argument("--output-dir", type=str, default="/data/docpc_project/models/colpali_pdfa_finetuned")
    p.add_argument("--pretrained-model-name-or-path", type=str, default="/data/docpc_project/models/colqwen2.5_v0.2_merged")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"])
    p.add_argument("--loss", type=str, default="pairwise",
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
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--split-ratio", type=float, default=0.02)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--dataloader-num-workers", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

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

    if args.loss in ["multi_positive", "combined"]:
        collator = MultiPositiveVisualRetrieverCollator(
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
            dataloader_drop_last=True,
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

    sample = train_dataset[0]
    n_pos = len(sample.get("pos_target", []))
    logger.info(f"Sample check: query={'query' in sample}; n_pos={n_pos}")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    if dist.is_initialized():
        dist.barrier()

    if args.trainer == "hf":
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
        logger.info(f"Merged model saved: {merge_model_path}")
