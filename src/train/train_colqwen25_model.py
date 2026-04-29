import argparse
import shutil
from pathlib import Path
import os
import json
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss, 
    ColbertPairwiseCELoss,
    CombinedMultiPositiveLoss,
    MultiPositiveInfoNCELoss,
)
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set
from colpali_engine.utils.logger_config import setup_logger
# from colpali_engine.utils.enhanced_model_eval import evaluate_model
from colpali_engine.utils.local_model_merge import model_merge
from colpali_engine.collators.visual_retriever_collator import (
    VisualRetrieverCollator,
    MultiPositiveVisualRetrieverCollator,
    AdaptiveMultiPositiveCollator,
)

logger = setup_logger("ColPali")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str,  default="/data/docpc_project/models/colqwen_finetuned", help="where to write model + script copy")
    p.add_argument("--dataset-name", type=str, default="/data/docpc_project/datasets/gold_datasets", help="dataset name")
    p.add_argument("--eval-image-dir", type=str, default=None, help="eval image dir")
    p.add_argument("--eval-dataset-path", type=str, default=None, help="eval dataset path")
    p.add_argument("--pretrained-model-name-or-path", type=str, default="/data/docpc_project/models/colqwen2.5_v0.2_merged", help="model name")
    p.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    p.add_argument("--tau", type=float, default=0.01, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="combined", 
                   choices=["ce", "pairwise", "multi_positive", "combined"], help="loss function to use")
    p.add_argument("--use-multi-positive", action="store_true", help="use multi-positive collator")
    p.add_argument("--max-positives", type=int, default=3, help="max positives per query")
    p.add_argument("--infonce-weight", type=float, default=1.0, help="weight for infonce loss")
    p.add_argument("--listwise-weight", type=float, default=0.1, help="weight for listwise loss")
    p.add_argument("--listwise-freq", type=int, default=5, help="frequency for listwise loss computation")
    p.add_argument("--pos-target-column", type=str, default="pos_target", 
                   help="column name for positive targets (auto-detect if None)")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")
    p.add_argument("--peft-r", type=int, default=32, help="PEFT r")
    p.add_argument("--peft-lora-alpha", type=int, default=32, help="PEFT lora_alpha")
    p.add_argument("--peft-lora-dropout", type=float, default=0.1, help="PEFT lora_dropout")
    p.add_argument("--peft-init-lora-weights", type=str, default="gaussian", help="PEFT init_lora_weights")
    p.add_argument("--peft-bias", type=str, default="none", help="PEFT bias")
    p.add_argument("--peft-task-type", type=str, default="FEATURE_EXTRACTION", help="PEFT task_type")
    p.add_argument("--peft-target-modules", type=str, default="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)", help="PEFT target_modules")

    p.add_argument("--num-train-epochs", type=int, default=5, help="number of training epochs")
    p.add_argument("--per-device-train-batch-size", type=int, default=32, help="per_device_train_batch_size")
    p.add_argument("--save-steps", type=int, default=500, help="save_steps")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="warmup_ratio")
    p.add_argument("--save-total-limit", type=int, default=1, help="save_total_limit")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="max_grad_norm")
    
    # 新增参数：验证集切分比例和评估频率
    p.add_argument("--split-ratio", type=float, default=0.0, help="validation set split ratio (0.0-1.0), 0.0 means no split")
    p.add_argument("--eval-steps", type=int, default=100, help="evaluation frequency in steps")

    return p.parse_args()   


if __name__ == "__main__":
    args = parse_args()

    # 根据损失函数类型选择
    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=False,
        )
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
        raise ValueError(f"Unknown loss function: {args.loss}")

    # 处理器配置
    # processor = ColQwen2_5_Processor.from_pretrained(
    #     pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    #     max_num_visual_tokens=768,
    # )
    processor = ColPaliProcessor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        max_num_visual_tokens=768,
    )

    # 根据是否使用多正例选择collator
    if args.use_multi_positive or args.loss in ["multi_positive", "combined"]:
        logger.info(f"🔄 使用多正例训练，最大正例数: {args.max_positives}")
        collator = MultiPositiveVisualRetrieverCollator(
            processor=processor,
            max_length=256,
            max_positives_per_query=args.max_positives,
            positive_sampling_strategy="random",
        )
    else:
        logger.info("🔄 使用传统单正例训练")
        collator = VisualRetrieverCollator(
            processor=processor,
            max_length=256,
        )

    # 加载数据集，支持自动切分验证集
    dataset_result = load_train_set(args.dataset_name, pos_target_column_name=args.pos_target_column, split_ratio=args.split_ratio)
    
    if args.split_ratio > 0.0:
        # 有验证集
        train_dataset, eval_dataset = dataset_result
        run_eval = True
        eval_strategy = "steps"
        logger.info(f"✅ 已自动切分验证集，比例: {args.split_ratio:.1%}")
    else:
        # 无验证集
        train_dataset = dataset_result
        eval_dataset = None
        run_eval = False
        eval_strategy = "no"
        logger.info("ℹ️  未启用验证集评估")

    model=ColPali.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        # use_cache=False,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.config.use_cache = False

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        # model=ColQwen2_5.from_pretrained(
        #     pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        #     torch_dtype=torch.bfloat16,
        #     use_cache=False,
        #     attn_implementation="flash_attention_2",
        # ).to("cuda"),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run_eval=run_eval,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size, 
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=args.per_device_train_batch_size,
            eval_strategy=eval_strategy,
            dataloader_num_workers=8,
            max_grad_norm=args.max_grad_norm,
            save_steps=args.save_steps,
            logging_steps=5,
            eval_steps=args.eval_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.lr,
            save_total_limit=args.save_total_limit,
            # 新增：保存最佳模型
            load_best_model_at_end=True if run_eval else False,
            metric_for_best_model="eval_loss" if run_eval else None,
            greater_is_better=False if run_eval else None,
        ),
        peft_config=LoraConfig(
            r=args.peft_r,
            lora_alpha=args.peft_lora_alpha,
            lora_dropout=args.peft_lora_dropout,
            init_lora_weights=args.peft_init_lora_weights,
            bias=args.peft_bias,
            task_type=args.peft_task_type,
            target_modules=args.peft_target_modules,
        )
        if args.peft
        else None,
    )
    # 手动设置collator以覆盖默认设置
    config._collator = collator  # 添加这个属性供trainer使用

    # 验证数据格式与训练模式匹配
    logger.info(f"\n🔍 数据格式验证:")
    train_dataset = config.train_dataset
    sample = train_dataset[0]
    
    if "pos_target" in sample and isinstance(sample["pos_target"], list):
        num_positives = len(sample["pos_target"])
        logger.info(f"  ✅ 检测到多正例数据，样本包含 {num_positives} 个正例")
        
        if args.loss in ["ce", "pairwise"]:
            logger.info(f"  ⚠️  警告: 使用单正例损失函数 '{args.loss}' 但数据包含多个正例")
            logger.info(f"  💡 建议: 使用 '--loss multi_positive' 或 '--loss combined'")
        
        if num_positives > args.max_positives:
            logger.info(f"  ⚠️  警告: 数据包含 {num_positives} 个正例，但 max_positives 设置为 {args.max_positives}")
            logger.info(f"  💡 建议: 增加 '--max-positives {num_positives}' 或更多")
    else:
        logger.info(f"  ✅ 检测到单正例数据")
        if args.loss in ["multi_positive", "combined"]:
            logger.info(f"  ⚠️  警告: 使用多正例损失函数但数据为单正例格式")
            logger.info(f"  💡 建议: 使用 '--loss ce' 或准备多正例数据")

    # make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    logger.info(f"📊 训练配置:")
    logger.info(f"  - 损失函数: {args.loss}")
    logger.info(f"  - 多正例训练: {args.use_multi_positive or args.loss in ['multi_positive', 'combined']}")
    logger.info(f"  - 学习率: {args.lr}")
    logger.info(f"  - 温度参数: {args.tau}")
    if args.loss == "combined":
        logger.info(f"  - Listwise权重: {args.listwise_weight}")
        logger.info(f"  - Listwise频率: {args.listwise_freq}")
    
    # 验证集配置信息
    if args.split_ratio > 0.0:
        logger.info(f"  - 验证集比例: {args.split_ratio:.1%}")
        logger.info(f"  - 评估频率: 每 {args.eval_steps} 步")
        logger.info(f"  - 保存最佳模型: 是 (基于eval_loss)")
    else:
        logger.info(f"  - 验证集评估: 未启用")

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    
    # 如果有自定义collator，手动设置
    if hasattr(config, '_collator'):
        trainer.collator = config._collator

    # 训练参数保存
    logger.info(f"config: {config}")
    logger.info(f"训练参数: {json.dumps(args.__dict__, indent=4)}")
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    trainer.train()
    trainer.save()
    # merge_model_path = os.path.join(args.output_dir, "sft-colqwen-merged")
    merge_model_path = os.path.join(args.output_dir, "sft-colpali-merged")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = model_merge(args.output_dir, merge_model_path, device)

    # if args.eval_dataset_path is not None and args.eval_image_dir is not None:
    #     evaluate_model(output_path, args.eval_image_dir, args.eval_dataset_path, output_path)


