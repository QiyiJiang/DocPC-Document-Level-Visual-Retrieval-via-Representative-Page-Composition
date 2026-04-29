#!/usr/bin/env bash
# Jina-CLIP 本地模型评估脚本：多策略 × 7 类别
#
# 使用 nohup 后台运行:
#   cd /data/docpc_project && nohup bash eval_jina_clip.sh > eval_jina_clip.log 2>&1 &
#   tail -f eval_jina_clip.log

set -e

# 强制离线，避免加载本地模型时仍访问 Hugging Face
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ==================== 参数配置 ====================
QUERY_ROOT="/data/docpc_project/dataset_generate_new/pdfa_test"
RESULTS_DIR="/data/docpc_project/500——eval_results/jina_clip"
POS_TARGET_COLUMN="pos_target_for_deepseek"
# 是否全库检索：true=索引 image_dir 下全部图片；false=只索引正例文档
FULL_POOL=true
BATCH_SIZE=32
TRUNCATE_DIM=512
GPU_ID=7
JINA_MODEL_PATH="/data/docpc_project/models/jina-clip-v2"
PYTHON_BIN="python"

CATEGORIES=(biology education finance government industrial legal research)
STRATEGIES=(first4)
# =================================================

mkdir -p "${RESULTS_DIR}"

if [ -n "${GPU_ID}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

echo "=============================================="
echo "Jina-CLIP 本地模型评估"
echo "  模型路径: ${JINA_MODEL_PATH}"
echo "  Python: ${PYTHON_BIN}"
echo "  GPU: ${GPU_ID}"
echo "  类别: ${CATEGORIES[*]}"
echo "  策略: ${STRATEGIES[*]}"
echo "  结果目录: ${RESULTS_DIR}"
echo "=============================================="

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "############## 策略: ${strategy} ##############"

    for cat in "${CATEGORIES[@]}"; do
        QUERY_JSON="${QUERY_ROOT}/${cat}/query_list_text_${cat}_with_pos_target_for_deepseek.json"
        IMAGE_DIR="${QUERY_ROOT}/${cat}/pos_target_for_deepseek_images_${strategy}"
        RUN_TAG="jina_clip__${strategy}__${cat}"

        if [ ! -f "${QUERY_JSON}" ]; then
            echo "  [${cat}] 跳过: query json 不存在"
            continue
        fi
        if [ ! -d "${IMAGE_DIR}" ]; then
            echo "  [${cat}] 跳过: 图片目录不存在 ${IMAGE_DIR}"
            continue
        fi

        echo "  [${cat}] 开始 → ${RUN_TAG}"

        EXTRA_ARGS=()
        [ "${FULL_POOL}" = "true" ] && EXTRA_ARGS+=(--full-pool)
        "${PYTHON_BIN}" eval_jina_clip.py \
            --model-path "${JINA_MODEL_PATH}" \
            --image-dir "${IMAGE_DIR}" \
            --eval-dataset-path "${QUERY_JSON}" \
            --results-dir "${RESULTS_DIR}" \
            --pos-target-column "${POS_TARGET_COLUMN}" \
            --run-tag "${RUN_TAG}" \
            --batch-size "${BATCH_SIZE}" \
            --truncate-dim "${TRUNCATE_DIM}" \
            "${EXTRA_ARGS[@]}"

        echo "  [${cat}] 完成"
    done

    echo "############## 策略 ${strategy} 全部类别完成 ##############"
done

echo ""
echo "=============================================="
echo "全部评估完成！结果在: ${RESULTS_DIR}"
echo "=============================================="
