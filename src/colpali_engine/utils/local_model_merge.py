#!/usr/bin/env python3
"""
手动合并 ColQwen2.5 LoRA 适配器到基础模型
将 LoRA 权重合并到基础模型中，生成完整的合并后模型
"""

import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel, PeftConfig
from safetensors.torch import load_file
import argparse
from tqdm import tqdm

from colpali_engine.utils.logger_config import setup_logger
logger = setup_logger("ColPali")

def load_adapter_config(adapter_path):
    """加载适配器配置"""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def merge_lora_weights(base_model, adapter_path, device="cuda"):
    """
    手动合并 LoRA 权重到基础模型
    """
    logger.info("🔄 开始合并 LoRA 权重...")
    
    # 加载适配器配置
    adapter_config = load_adapter_config(adapter_path)
    base_model_name = adapter_config["base_model_name_or_path"]
    
    logger.info(f"📂 基础模型: {base_model_name}")
    logger.info(f"📂 适配器路径: {adapter_path}")
    
    # 使用 PEFT 加载模型和适配器
    logger.info("📥 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    logger.info("📥 加载 LoRA 适配器...")
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    logger.info("🔗 合并权重...")
    # 合并 LoRA 权重到基础模型
    merged_model = model_with_adapter.merge_and_unload()
    
    return merged_model

def save_merged_model(merged_model, tokenizer, processor, output_path):
    """保存合并后的模型"""
    logger.info(f"💾 保存合并后的模型到: {output_path}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # 使用 safetensors 格式
        max_shard_size="5GB"
    )
    
    # 保存 tokenizer
    if tokenizer:
        tokenizer.save_pretrained(output_path)
    
    # 保存 processor
    if processor:
        processor.save_pretrained(output_path)
    
    logger.info("✅ 模型保存完成!")

def manual_merge_alternative(adapter_path, output_path, device="cuda"):
    """
    替代方案: 直接使用 PEFT 的合并功能
    """
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from colpali_engine.models import ColPali, ColPaliProcessor
    
    logger.info("🔄 使用 PEFT 自动合并方案...")
    
    # 加载配置
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # 加载基础模型
    logger.info("📥 加载基础模型...")
    base_model = ColQwen2_5.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    # base_model = ColPali.from_pretrained(
    #     peft_config.base_model_name_or_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map=device,
    #     trust_remote_code=True
    # )
    # 加载带适配器的模型
    logger.info("📥 加载 LoRA 适配器...")
    model_with_lora = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    # 合并权重
    logger.info("🔗 合并权重...")
    merged_model = model_with_lora.merge_and_unload()
    
    # 加载处理器
    logger.info("📥 加载处理器...")
    try:
        processor = ColQwen2_5_Processor.from_pretrained(adapter_path)
        # processor = ColPaliProcessor.from_pretrained(adapter_path)

    except:
        processor = ColQwen2_5_Processor.from_pretrained(peft_config.base_model_name_or_path)
        # processor = ColPaliProcessor.from_pretrained(peft_config.base_model_name_or_path)

    
    # 保存合并后的模型
    save_merged_model(merged_model, None, processor, output_path)
    
    return merged_model, processor

def verify_merged_model(model_path, adapter_path):
    """验证合并后的模型"""
    logger.info("🔍 验证合并后的模型...")
    
    try:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from colpali_engine.models import ColPali, ColPaliProcessor
        
        # 加载合并后的模型
        merged_model = ColQwen2_5.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )

        # merged_model = ColPali.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map="cuda:0"
        # )
        
        processor = ColQwen2_5_Processor.from_pretrained(model_path)
        # processor = ColPaliProcessor.from_pretrained(model_path)

        
        logger.info("✅ 合并后的模型加载成功!")
        logger.info(f"📊 模型参数数量: {sum(p.numel() for p in merged_model.parameters()):,}")
        
        return True, merged_model, processor
        
    except Exception as e:
        logger.info(f"❌ 验证失败: {e}")
        return False, None, None

def model_merge(adapter_path, output_path, device="cuda", verify=False):
    """合并模型"""
    try:
        # 检查输入路径
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"适配器路径不存在: {adapter_path}")
        
        # 检查必要文件
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        for file in required_files:
            file_path = os.path.join(adapter_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"缺少必要文件: {file_path}")
        
        # 执行合并
        merged_model, processor = manual_merge_alternative(
            adapter_path, 
            output_path, 
            device
        )
        
        # 验证合并结果
        if verify:
            success, verified_model, verified_processor = verify_merged_model(
                output_path, 
                adapter_path
            )
            if success:
                logger.info("🎉 模型合并并验证成功!")
            else:
                logger.info("⚠️ 模型合并完成但验证失败")
        else:
            logger.info("🎉 模型合并完成!")
        
        return output_path
        
    except Exception as e:
        logger.info(f"❌ 合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="手动合并 ColQwen2.5 LoRA 适配器")
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default="/data/docpc_project/models/colqwen2.5_pdfa_all",
        help="LoRA 适配器路径"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="/data/docpc_project/models/colqwen2.5_pdfa_all/sft-colqwen-merged",
        help="输出合并后模型的路径"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="设备类型"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="验证合并后的模型"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 ColQwen2.5 LoRA 模型合并工具")
    logger.info("=" * 50)
    
    model_merge(args.adapter_path, args.output_path, args.device, args.verify)

if __name__ == "__main__":
    main()