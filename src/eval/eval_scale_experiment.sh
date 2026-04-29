#!/usr/bin/env bash
# 规模梯度实验：对比基线 vs 微调模型在不同文档规模下的表现
#
# 用法:
#   cd /data/docpc_project && nohup bash eval_scale_experiment.sh > eval_scale_experiment.log 2>&1 &
#   tail -f eval_scale_experiment.log

set -e

# ==================== 参数配置 ====================
GPU_ID=1

# ── 模型路径（切换基线/微调模型时修改这里） ──
MODEL_PATH="/data/docpc_project/models/colqwen_pdfa_all_first4/sft-colqwen-merged"
# MODEL_PATH="/path/to/baseline_model"

QUERY_ROOT="/data/docpc_project/dataset_generate_new/pdfa_test"
# 图片目录模板，{cat} 会被替换为类别名
IMAGE_DIR_TEMPLATE="${QUERY_ROOT}/{cat}/pos_target_for_deepseek_images_first4"

# 结果目录（基线和微调模型用不同目录，方便对比）
RESULTS_DIR="/data/docpc_project/eval_results/scale_experiment_qwen——2"
# RESULTS_DIR="/data/docpc_project/eval_results/scale_experiment_baseline"

POS_TARGET_COLUMN="pos_target_for_deepseek"

# 采样比例
SAMPLE_RATIOS="0.2 0.4 0.6 0.8 1.0"

# 随机种子（基线和微调模型必须相同，确保采样同一子集）
SEED=42

CATEGORIES=(biology education finance government industrial legal research)
# ==================== 参数配置结束 ====================

MODEL_NAME=$(basename "$(dirname "${MODEL_PATH}")")
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=============================================="
echo "规模梯度实验: ${#CATEGORIES[@]} 个类别"
echo "模型: ${MODEL_PATH}"
echo "采样比例: ${SAMPLE_RATIOS}"
echo "种子: ${SEED}"
echo "GPU: ${GPU_ID}"
echo "结果目录: ${RESULTS_DIR}"
echo "=============================================="

for cat in "${CATEGORIES[@]}"; do
    QUERY_JSON="${QUERY_ROOT}/${cat}/query_list_text_${cat}_with_pos_target_for_deepseek.json"
    IMAGE_DIR="${IMAGE_DIR_TEMPLATE//\{cat\}/${cat}}"

    if [ ! -f "${QUERY_JSON}" ]; then
        echo "[${cat}] 跳过: ${QUERY_JSON} 不存在"
        continue
    fi
    if [ ! -d "${IMAGE_DIR}" ]; then
        echo "[${cat}] 跳过: ${IMAGE_DIR} 不存在"
        continue
    fi

    echo ""
    echo "====== [${cat}] 开始规模梯度实验 ======"
    echo "  Query JSON: ${QUERY_JSON}"
    echo "  Image DIR:  ${IMAGE_DIR}"

    python eval_scale_experiment.py \
        --model-path "${MODEL_PATH}" \
        --image-dir "${IMAGE_DIR}" \
        --eval-dataset-path "${QUERY_JSON}" \
        --results-dir "${RESULTS_DIR}" \
        --pos-target-column "${POS_TARGET_COLUMN}" \
        --sample-ratios ${SAMPLE_RATIOS} \
        --seed "${SEED}" \
        --gpu-id "${GPU_ID}" \
        --run-tag "${cat}"

    echo "====== [${cat}] 完成 ======"
done

echo ""
echo "=============================================="
echo "全部类别规模梯度实验完成！结果在: ${RESULTS_DIR}"
echo "=============================================="
