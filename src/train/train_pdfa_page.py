"""
Page-level PDFA 训练：支持 ColQwen2.5 / ColPali，通过 --model-type 切换，线性串行微调。
数据约定（expanded_x4 JSON + 单页图片）:
  - json_root/{cat}/query_list_text_{cat}_with_pos_target_for_deepseek_cleaned_expanded_x4.json
  - image_base/{cat}_{image_suffix}/{stem}.png
  - JSON 每条: {"query": "...", "pos_target": "3427061_0.txt"}
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
from peft import LoraConfig, PeftModel, PeftConfig
from transformers import TrainingArguments, TrainerCallback
from PIL import Image
from typing import Any, Dict, List

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.gpu_stats import print_summary
from colpali_engine.utils.logger_config import setup_logger
from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator

logger = setup_logger("PageTrain")

# ---------------------------------------------------------------------------
# 模型注册表：通过 model_type 选择 model / processor 类
# ---------------------------------------------------------------------------
MODEL_CLASSES = {
    "colqwen": (ColQwen2_5, ColQwen2_5_Processor),
    "colpali": (ColPali, ColPaliProcessor),
}

CATEGORIES = [
    "biology", "education", "finance", "government",
    "industrial", "legal", "research",
]

JSON_TEMPLATE = "query_list_text_{cat}_with_pos_target_for_deepseek_cleaned_expanded_x4.json"


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.6e}"
        return f"{v:.6f}"
    return str(v)


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not is_main_process():
            return
        parts = [f"{k}={_fmt(v)}" for k, v in sorted(logs.items())]
        print("\n  [train] " + "  ".join(parts), flush=True)
        sys.stdout.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        if is_main_process():
            print("\n  [train] Training loop started.\n", flush=True)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Corpus：从 doc_id 解析到图片路径
# ---------------------------------------------------------------------------
class ImagePageCorpus:
    """
    doc_id: "biology/3427061_0.txt"
    image:  image_base / biology_pos_target_for_deepseek_images_first4 / 3427061_0.png
    """
    def __init__(self, image_base: Path, image_suffix: str):
        self.image_base = Path(image_base)
        self.image_suffix = image_suffix

    def retrieve(self, doc_id: str) -> Image.Image:
        if "/" in doc_id:
            cat, filename = doc_id.split("/", 1)
        else:
            cat, filename = "", doc_id
        stem = Path(filename).stem
        if cat:
            img_dir = self.image_base / f"{cat}_{self.image_suffix}"
        else:
            img_dir = self.image_base / self.image_suffix
        path = img_dir / f"{stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path} (doc_id={doc_id})")
        return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_page_train_set(
    json_root: str | Path,
    image_base: str | Path,
    image_suffix: str,
    categories: list[str] | None = None,
    split_ratio: float = 0.0,
    seed: int = 42,
):
    json_root = Path(json_root)
    categories = categories or CATEGORIES

    logger.info(f"JSON root: {json_root}")
    logger.info(f"Image base: {image_base}")
    logger.info(f"Categories: {categories}")

    corpus = ImagePageCorpus(Path(image_base), image_suffix)
    all_ds = []

    for cat in categories:
        json_path = json_root / cat / JSON_TEMPLATE.format(cat=cat)
        if not json_path.exists():
            logger.warning(f"JSON not found, skip: {json_path}")
            continue
        ds = load_dataset("json", data_files={"train": str(json_path)}, split="train")

        def map_fn(row, _cat=cat):
            raw = row["pos_target"]
            if not isinstance(raw, list):
                raw = [raw]
            return {"pos_target": [f"{_cat}/{x}" for x in raw if x]}

        ds = ds.map(map_fn, num_proc=1, desc=f"[{cat}] prefix")
        ds = ds.filter(lambda r: len(r["pos_target"]) > 0, num_proc=1, desc=f"[{cat}] filter")
        all_ds.append(ds)
        logger.info(f"  {cat}: {len(ds)} samples")

    if not all_ds:
        raise ValueError("No valid data found")

    full = concatenate_datasets(all_ds)
    logger.info(f"Total: {len(full)} samples")

    if split_ratio > 0:
        full = full.shuffle(seed=seed)
        n = len(full)
        eval_n = int(n * split_ratio)
        train_n = n - eval_n
        logger.info(f"Train/Eval: {train_n} / {eval_n}")
        return (
            ColPaliEngineDataset(full.select(range(train_n)), corpus=corpus, pos_target_column_name="pos_target"),
            ColPaliEngineDataset(full.select(range(train_n, n)), corpus=corpus, pos_target_column_name="pos_target"),
        )
    return ColPaliEngineDataset(full, corpus=corpus, pos_target_column_name="pos_target")


# ---------------------------------------------------------------------------
# 模型合并（根据 model_type 选类）
# ---------------------------------------------------------------------------
def merge_model(adapter_path: str, output_path: str, model_type: str, device: str = "cuda"):
    ModelClass, ProcessorClass = MODEL_CLASSES[model_type]
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_path = peft_config.base_model_name_or_path

    logger.info(f"Merge [{model_type}]: base={base_path}, adapter={adapter_path}")

    base_model = ModelClass.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
    merged = model_with_lora.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")

    try:
        proc = ProcessorClass.from_pretrained(adapter_path)
    except Exception:
        proc = ProcessorClass.from_pretrained(base_path)
    proc.save_pretrained(output_path)

    logger.info(f"Merged model saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Page-level PDFA training (ColQwen / ColPali)")
    p.add_argument("--model-type", type=str, required=True, choices=["colqwen", "colpali"],
                   help="colqwen → ColQwen2_5; colpali → ColPali")
    p.add_argument("--pretrained-model-name-or-path", type=str, required=True)
    p.add_argument("--json-root", type=str, required=True,
                   help="JSON root: {json_root}/{cat}/query_list_text_{cat}_..._expanded_x4.json")
    p.add_argument("--image-base", type=str, required=True,
                   help="Image base: {image_base}/{cat}_{image_suffix}/{stem}.png")
    p.add_argument("--image-suffix", type=str, default="pos_target_for_deepseek_images_first4")
    p.add_argument("--categories", type=str, nargs="+", default=None)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-train-epochs", type=int, default=5)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--split-ratio", type=float, default=0.02)
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--dataloader-num-workers", type=int, default=8)

    p.add_argument("--peft", action="store_true")
    p.add_argument("--peft-r", type=int, default=32)
    p.add_argument("--peft-lora-alpha", type=int, default=32)
    p.add_argument("--peft-lora-dropout", type=float, default=0.1)
    p.add_argument("--peft-init-lora-weights", type=str, default="gaussian")
    p.add_argument("--peft-bias", type=str, default="none")
    p.add_argument("--peft-task-type", type=str, default="FEATURE_EXTRACTION")
    p.add_argument("--peft-target-modules", type=str,
                   default=r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    ModelClass, ProcessorClass = MODEL_CLASSES[args.model_type]

    # ---- Loss: pairwise ----
    loss_func = ColbertPairwiseCELoss(normalize_scores=False)

    # ---- Processor ----
    proc_kwargs = {}
    if args.model_type == "colqwen":
        proc_kwargs["max_num_visual_tokens"] = 768
    processor = ProcessorClass.from_pretrained(args.pretrained_model_name_or_path, **proc_kwargs)

    # ---- Collator ----
    collator = VisualRetrieverCollator(processor=processor, max_length=256)

    # ---- Data ----
    ds_result = load_page_train_set(
        json_root=args.json_root,
        image_base=args.image_base,
        image_suffix=args.image_suffix,
        categories=args.categories,
        split_ratio=args.split_ratio,
    )

    if args.split_ratio > 0:
        train_ds, eval_ds = ds_result
        run_eval = True
        eval_strategy = "steps"
    else:
        train_ds = ds_result
        eval_ds = None
        run_eval = False
        eval_strategy = "no"

    # ---- Model ----
    model = ModelClass.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.config.use_cache = False

    # ---- Training Config ----
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
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

    # ---- Sample check ----
    sample = train_ds[0]
    logger.info(f"Sample check: query={'query' in sample}, pos_target={'pos_target' in sample}")

    # ---- Save script & args ----
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    if dist.is_initialized():
        dist.barrier()

    # ---- Train ----
    class _TrainerWithLog(ColModelTraining):
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
                print(f"\n  [train] model_type={args.model_type}, starting...\n", flush=True)
            if dist.is_initialized():
                dist.barrier()
            result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
            print_summary(result)

    t = _TrainerWithLog(config)
    t.collator = config._collator
    t.train()
    t.save()

    # ---- Merge ----
    if is_main_process():
        merge_path = os.path.join(args.output_dir, f"sft-{args.model_type}-merged")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        merge_model(args.output_dir, merge_path, args.model_type, device)
        logger.info(f"Done! Merged → {merge_path}")
