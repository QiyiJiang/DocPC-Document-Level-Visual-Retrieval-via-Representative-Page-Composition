#!/usr/bin/env bash
# 页面级检索 + 策略聚合评估：支持多模型 × 多策略组合，7 个类别循环。
#
# 使用 nohup 后台运行:
#   cd /data/docpc_project && nohup bash eval_page_strategy.sh > eval_page_strategy.log 2>&1 &
#   tail -f eval_page_strategy.log

set -e

# ==================== 全局参数 ====================
# GPU_ID: 卡号，写 1 用 1 号卡，写 2 用 2 号卡（脚本内会转为 CUDA 0-based 索引）
GPU_ID=7
QUERY_ROOT="/data/docpc_project/dataset_generate_new/pdfa_test"
METADATA_PATH_TEMPLATE="/data/docpc_project/datasets/pdfa_test/image_page_metadata_{cat}.json"
IMAGE_DIR_TEMPLATE="/data/docpc_project/datasets/pdfa_test/image_page_{cat}"
RESULTS_DIR="/data/docpc_project/500——eval_results/page_strategy"
POS_TARGET_COLUMN="pos_target_for_deepseek"
SCORE_AGG="max"
# 是否全库检索：true=编码 metadata 全部页面并在全库上检索；false=只编码正例文档页面（默认）
FULL_POOL="true"
# 检索池文档数（留空则按上面 FULL_POOL）。设为整数 N 时：N<正例文档数→用正例数，N>全量文档数→用全量，否则每类用 N 个文档作为检索池
POOL_SIZE=
CATEGORIES=(biology education finance government industrial legal research)
ALL_STRATEGIES=(first4 first9 last4 uniform4 random4 first2_last2 all_pages)

# ==========================================================
#  ★★★ 在这里配置: 模型 × 策略 ★★★
#
#  格式: "模型路径|模型类型|策略1,策略2,..."
#    - 模型类型: colpali 或 colqwen
#    - 策略用逗号分隔; 写 all 表示全部6种
#    - 可用策略: first4  first9  last4  uniform4  random4  first2_last2  all_pages
#
#  示例:
#    "/.../sft-colpali-merged|colpali|first4"           → 只跑 first4
#    "/.../sft-colpali-merged|colpali|first4,last4"     → 跑两种
#    "/.../colpali_v1.3_merged|colpali|all"             → 跑全部6种
# ==========================================================
CONFIGS=(
    # # ---------- ColPali 微调模型 ----------
    # "/data/docpc_project/models/colpali_pdfa_all_first4/sft-colpali-merged|colpali|all"
    # "/data/docpc_project/models/colpali_pdfa_all_first2_last2/sft-colpali-merged|colpali|all"
    # "/data/docpc_project/models/colpali_pdfa_all_random4/sft-colpali-merged|colpali|all"
    # "/data/docpc_project/models/colpali_pdfa_all_uniform4/sft-colpali-merged|colpali|all"

    # ---------- ColPali 基线模型 ----------
    # "/data/docpc_project/models/colpali_v1.3_merged|colpali|all"
    # "/data/docpc_project/models/colpali_page_first4/sft-colpali-merged|colpali|all"

    # # ---------- ColQwen 微调模型 ----------
    # "/data/docpc_project/models/colqwen_pdfa_all_first2_last2/sft-colqwen-merged|colqwen|all"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4/sft-colqwen-merged|colqwen|all"
    # "/data/docpc_project/models/colqwen_pdfa_all_last4/sft-colqwen-merged|colqwen|all"
    # "/data/docpc_project/models/colqwen_pdfa_all_random4/sft-colqwen-merged|colqwen|all"
    # "/data/docpc_project/models/colqwen_pdfa_all_uniform4/sft-colqwen-merged|colqwen|all"

    # ---------- ColQwen 基线模型 ----------
    "/data/docpc_project/models/colqwen2.5_v0.2_merged|colqwen|first9"
    "/data/docpc_project/models/colqwen_page_first4/sft-colqwen-merged|colqwen|first9"
)
# ==========================================================

# 转为 0-based：1号卡→0，2号卡→1
export CUDA_VISIBLE_DEVICES=$((${GPU_ID} - 1))
mkdir -p "${RESULTS_DIR}"

total_configs=${#CONFIGS[@]}
echo "=============================================="
echo "页面级策略聚合评估: ${total_configs} 个模型配置"
echo "类别: ${CATEGORIES[*]}"
echo "GPU: ${GPU_ID}"
echo "=============================================="

config_idx=0
for config_line in "${CONFIGS[@]}"; do
    config_idx=$((config_idx + 1))

    IFS='|' read -r MODEL_PATH MODEL_TYPE STRATEGIES_STR <<< "${config_line}"

    # 微调模型取倒数第二级目录；基座模型取最后一级目录，避免重名覆盖
    PARENT_DIR=$(basename "$(dirname "${MODEL_PATH}")")
    if [ "${PARENT_DIR}" = "models" ]; then
        MODEL_NAME=$(basename "${MODEL_PATH}")
    else
        MODEL_NAME="${PARENT_DIR}"
    fi

    # 解析策略列表
    if [ "${STRATEGIES_STR}" = "all" ]; then
        STRATEGIES=("${ALL_STRATEGIES[@]}")
    else
        IFS=',' read -ra STRATEGIES <<< "${STRATEGIES_STR}"
    fi

    echo ""
    echo "##############################################"
    echo "# [${config_idx}/${total_configs}] 模型: ${MODEL_NAME}"
    echo "#   路径: ${MODEL_PATH}"
    echo "#   类型: ${MODEL_TYPE}"
    echo "#   策略: ${STRATEGIES[*]}"
    echo "##############################################"

    for cat in "${CATEGORIES[@]}"; do
        EVAL_DATASET="${QUERY_ROOT}/${cat}/query_list_text_${cat}_with_pos_target_for_deepseek.json"
        METADATA_PATH="${METADATA_PATH_TEMPLATE//\{cat\}/${cat}}"
        IMAGE_DIR="${IMAGE_DIR_TEMPLATE//\{cat\}/${cat}}"

        if [ ! -f "${EVAL_DATASET}" ]; then
            echo "  [${cat}] 跳过: ${EVAL_DATASET} 不存在"
            continue
        fi

        RUN_TAG="${MODEL_NAME}__${cat}"

        echo ""
        echo "  ====== [${cat}] 开始评估 ======"
        echo "    Query JSON:  ${EVAL_DATASET}"
        echo "    Metadata:    ${METADATA_PATH}"
        echo "    Image DIR:   ${IMAGE_DIR}"
        echo "    策略:        ${STRATEGIES[*]}"
        echo "    run_tag:     ${RUN_TAG}"

        EXTRA_ARGS=()
        [ "${FULL_POOL}" = "true" ] && EXTRA_ARGS+=(--full-pool)
        [ -n "${POOL_SIZE}" ] && EXTRA_ARGS+=(--pool-size "${POOL_SIZE}")
        python eval_colpali_page_strategy.py \
            --model-path "${MODEL_PATH}" \
            --model-type "${MODEL_TYPE}" \
            --eval-dataset-path "${EVAL_DATASET}" \
            --metadata-path "${METADATA_PATH}" \
            --image-dir "${IMAGE_DIR}" \
            --results-dir "${RESULTS_DIR}" \
            --pos-target-column "${POS_TARGET_COLUMN}" \
            --score-agg "${SCORE_AGG}" \
            --strategy ${STRATEGIES[@]} \
            --run-tag "${RUN_TAG}" \
            "${EXTRA_ARGS[@]}"

        echo "  ====== [${cat}] 评估完成 ======"
    done
done

echo ""
echo "=============================================="
echo "全部模型 × 策略组合评估完成！"
echo "结果目录: ${RESULTS_DIR}"
echo "=============================================="
