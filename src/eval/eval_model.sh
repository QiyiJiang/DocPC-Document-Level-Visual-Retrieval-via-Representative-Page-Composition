#!/usr/bin/env bash
# 批量评估脚本：支持多模型 × 多策略 × 多分辨率 × 7 个类别。
# 单模型单策略时在 CONFIGS 中只保留一行即可（等价于原 eval_all.sh）。
#
# 使用 nohup 后台运行:
#   cd /data/docpc_project && nohup bash eval_all_combo.sh > eval_all_combo.log 2>&1 &
#   tail -f eval_all_combo.log

set -e

# ==================== 全局参数 ====================
GPU_ID=6
QUERY_ROOT="/data/docpc_project/dataset_generate_new/pdfa_test"
RESULTS_DIR="/data/docpc_project/500——eval_results/combo"
POS_TARGET_COLUMN="pos_target_for_deepseek"
FULL_POOL=true
# 检索池大小（留空则按上面 FULL_POOL：true=全量，false=仅正例）
# 设为整数 N 时：N<正例数→用正例数，N>全量→用全量，否则 7 个类别均用 N 张图作为检索池
POOL_SIZE=
CATEGORIES=(biology education finance government industrial legal research)
ALL_STRATEGIES=(first4 first2_last2 last4 uniform4 random4 clip4 first1 first9 first16)

# 图片 resize 分辨率列表（WxH 格式，留空或 "original" 表示原图不 resize）
# 例: RESIZES=(128x166 256x331 512x662 original)
# 带 max_pixels 的格式: "WxH:MAX_PIXELS"，如 "1024x1325:1400000"
RESIZES=(1024x1325:1400000)

# 覆盖 processor 的 max_pixels（仅对 original 生效，留空则用模型默认值）
ORIGINAL_MAX_PIXELS=

# ==========================================================
#  ★★★ 在这里配置: 模型 × 策略 ★★★
#
#  格式: "模型路径 | 模型类型 | 策略1,策略2,..."
#    - 模型类型: colpali 或 colqwen
#    - 策略用逗号分隔; 写 all 表示全部5种
#    - 可用策略: first4  last4  uniform4  random4  first2_last2
#
#  示例:
#    "/.../sft-colpali-merged|colpali|first4"           → 只跑 first4
#    "/.../sft-colpali-merged|colpali|first4,last4"     → 跑两种
#    "/.../colpali_v1.3_merged|colpali|all"             → 跑全部5种
# ==========================================================
CONFIGS=(


    # # ---------- ColPali 基线模型 ----------
    # "/data/docpc_project/models/colpali_v1.3_merged|colpali|all"
    # "/data/docpc_project/models/colpali_page_first4/sft-colpali-merged|colpali|all"


    # ---------- ColPali 微调模型 ----------
    # "/data/docpc_project/models/colpali_pdfa_all_first4/sft-colpali-merged|colpali|all"
    # "/data/docpc_project/models/colpali_pdfa_all_first2_last2/sft-colpali-merged|colpali|all"
    # "/data/docpc_project/models/colpali_pdfa_all_random4/sft-colpali-merged|colpali|all"
    # "/data/docpc_project/models/colpali_pdfa_all_uniform4/sft-colpali-merged|colpali|all"




    # # # ---------- ColQwen 基线模型 ----------
    # "/data/docpc_project/models/colqwen2.5_v0.2_merged|colqwen|first9"
    # "/data/docpc_project/models/colqwen_page_first4/sft-colqwen-merged|colqwen|first9"

    # # ---------- ColQwen 微调模型 ----------
    # "/data/docpc_project/models/colqwen_pdfa_all_first2_last2/sft-colqwen-merged|colqwen|all"
    "/data/docpc_project/models/colqwen_pdfa_all_first4/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_last4/sft-colqwen-merged|colqwen|random4"
    # "/data/docpc_project/models/colqwen_pdfa_all_random4/sft-colqwen-merged|colqwen|first4,first2_last2,last4,uniform4,random4"
    # "/data/docpc_project/models/colqwen_pdfa_all_uniform4/sft-colqwen-merged|colqwen|first4,first2_last2,last4,uniform4,random4"



    # # ---------- 消融 freq ----------
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——1/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——3/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——7/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——9/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——11/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——13/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——15/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——20/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——30/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——40/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——50/sft-colqwen-merged|colqwen|first4"

    # # # ---------- 消融 loss function ----------
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——list/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_first4——multi/sft-colqwen-merged|colqwen|first4"

    # "/data/docpc_project/models/colqwen_pdfa_all_first16/sft-colqwen-merged|colqwen|first16"
    # "/data/docpc_project/models/colqwen_pdfa_all_first1/sft-colqwen-merged|colqwen|first1"
    # "/data/docpc_project/models/colqwen_pdfa_all_first9/sft-colqwen-merged|colqwen|first9"
    # "/data/docpc_project/models/colqwen_pdfa_all_first2—freq9/sft-colqwen-merged|colqwen|first4"

    # "/data/docpc_project/models/colqwen_pdfa_all_first4-40-cleaned/sft-colqwen-merged|colqwen|first4"
    # "/data/docpc_project/models/colqwen_pdfa_all_clip4/sft-colqwen-merged|colqwen|clip4"

)
# ==========================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

total_configs=${#CONFIGS[@]}
echo "=============================================="
echo "批量组合评估: ${total_configs} 个模型配置"
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

    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "  ---- 策略: ${strategy} ----"

        for resize_entry in "${RESIZES[@]}"; do
            # 解析 "WxH:MAX_PIXELS" 或 "WxH" 或 "original"
            RESIZE_ARGS=()
            RESIZE_TAG=""
            if [ "${resize_entry}" = "original" ] || [ -z "${resize_entry}" ]; then
                if [ -n "${ORIGINAL_MAX_PIXELS}" ]; then
                    RESIZE_ARGS=(--max-pixels "${ORIGINAL_MAX_PIXELS}")
                    RESIZE_TAG="__mp${ORIGINAL_MAX_PIXELS}"
                fi
                echo "    ---- 分辨率: 原图${ORIGINAL_MAX_PIXELS:+ (max_pixels=$ORIGINAL_MAX_PIXELS)} ----"
            else
                RES_PART="${resize_entry%%:*}"
                MP_PART="${resize_entry#*:}"
                if [ "${MP_PART}" = "${resize_entry}" ]; then
                    MP_PART=""
                fi
                RESIZE_ARGS=(--resize "${RES_PART}")
                RESIZE_TAG="__${RES_PART}"
                if [ -n "${MP_PART}" ]; then
                    RESIZE_ARGS+=(--max-pixels "${MP_PART}")
                    RESIZE_TAG="${RESIZE_TAG}__mp${MP_PART}"
                fi
                echo "    ---- 分辨率: ${RES_PART}${MP_PART:+ (max_pixels=$MP_PART)} ----"
            fi

            for cat in "${CATEGORIES[@]}"; do
                IMG_DIR="${QUERY_ROOT}/${cat}/pos_target_for_deepseek_images_${strategy}"
                QUERY_JSON="${QUERY_ROOT}/${cat}/query_list_text_${cat}_with_pos_target_for_deepseek.json"
                RUN_TAG="${MODEL_NAME}__${strategy}${RESIZE_TAG}__${cat}"

                if [ ! -d "${IMG_DIR}" ] || [ ! -f "${QUERY_JSON}" ]; then
                    echo "      [${cat}] 跳过: 目录或 query 文件不存在"
                    continue
                fi

                echo "      [${cat}] 开始评估..."

                POOL_ARGS=()
                if [ -n "${POOL_SIZE}" ]; then
                    POOL_ARGS+=(--pool-size "${POOL_SIZE}")
                fi
                python eval_model.py \
                    --model-path "${MODEL_PATH}" \
                    --model-type "${MODEL_TYPE}" \
                    --image-dir "${IMG_DIR}" \
                    --eval-dataset-path "${QUERY_JSON}" \
                    --results-dir "${RESULTS_DIR}" \
                    --use-milvus \
                    --pos-target-column "${POS_TARGET_COLUMN}" \
                    --run-tag "${RUN_TAG}" \
                    --full-pool \
                    "${POOL_ARGS[@]}" \
                    "${RESIZE_ARGS[@]}"

                echo "      [${cat}] 完成"
            done

            echo "    ---- 分辨率 ${resize_entry} 全部完成 ----"
        done

        echo "  ---- 策略 ${strategy} 全部完成 ----"
    done
done

echo ""
echo "=============================================="
echo "全部模型 × 策略组合评估完成！"
echo "结果目录: ${RESULTS_DIR}"
echo "=============================================="
