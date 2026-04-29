#!/usr/bin/env bash
# PDFA 多类别训练启动脚本（使用 train_pdfa_colpali.py，不修改原 train 代码）
# 数据约定：--dataset-root 下每类有 query_list_text_{cat}_with_pos_target_for_deepseek_cleaned.json 与 pos_target_for_deepseek_images_first4/
# 通过 N_GPUS 切换：1=单卡（python），>1=多卡（torchrun，每卡 per-device-train-batch-size 不变，全局 batch = N_GPUS × 8）
#
# 使用 nohup 后台运行（断线不中断）:
#   cd /data/docpc_project && nohup bash train_pdfa.sh > train_pdfa_last4.log 2>&1 &
# 查看日志: tail -f train_pdfa_last4.log

set -e
# ===== NCCL / 多卡稳定（与 docpc_project train_entry.sh 一致）=====
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

# ===== 卡数：1=单卡训练，4=四卡训练（全局 batch = 4×8=32） =====
N_GPUS=1
# 使用的 GPU 编号（单卡时用第一个，多卡时用逗号分隔的连续或指定卡）
CUDA_DEVICES="5"
# 多卡时 master 端口（避免多任务冲突）
MASTER_PORT=29503

PDFA_ROOT="/data/docpc_project/dataset_generate_new/pdfa"
OUT_BASE="/data/docpc_project/models"
MODEL_BASE="/data/docpc_project/models/colqwen2.5_v0.2_merged"

# -----------------------------------------------------------------------------
# 单类别示例：仅 biology
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python train_pdfa_colpali.py \
#     --dataset-root "${PDFA_ROOT}" \
#     --categories biology \
#     --output-dir "${OUT_BASE}/colpali_pdfa_biology" \
#     --pretrained-model-name-or-path "${MODEL_BASE}" \
#     --loss combined \
#     --use-multi-positive \
#     --max-positives 10 \
#     --peft \
#     --peft-r 32 \
#     --peft-lora-alpha 32 \
#     --peft-lora-dropout 0.1 \
#     --num-train-epochs 5 \
#     --per-device-train-batch-size 8 \
#     --split-ratio 0.02 \
#     --save-steps 500 \
#     --eval-steps 200 \
#     --save-total-limit 2

# -----------------------------------------------------------------------------
# 多类别：指定类别列表（biology education legal）
# -----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python train_pdfa_colpali.py \
#     --dataset-root "${PDFA_ROOT}" \
#     --categories biology education legal \
#     --output-dir "${OUT_BASE}/colpali_pdfa_3cat" \
#     --pretrained-model-name-or-path "${MODEL_BASE}" \
#     --loss combined \
#     --use-multi-positive \
#     --max-positives 10 \
#     --peft \
#     --num-train-epochs 5 \
#     --per-device-train-batch-size 8 \
#     --split-ratio 0.02 \
#     --save-steps 500 \
#     --eval-steps 200 \
#     --save-total-limit 2

# -----------------------------------------------------------------------------
# 多类别：first9（3×3 九宫格图）
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

_COMMON_ARGS=(
    --dataset-root "${PDFA_ROOT}"
    --pretrained-model-name-or-path "${MODEL_BASE}"
    --lr 5e-5
    --tau 0.01
    --loss combined
    --use-multi-positive
    --max-positives 10
    --infonce-weight 1
    --listwise-weight 0.1
    --listwise-freq 5
    --pos-target-column pos_target_for_deepseek
    --peft
    --peft-r 32
    --peft-lora-alpha 32
    --peft-lora-dropout 0.1
    --peft-init-lora-weights gaussian
    --peft-bias none
    --peft-task-type FEATURE_EXTRACTION
    --peft-target-modules "(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*\$|.*(custom_text_proj).*\$)"
    --num-train-epochs 5
    --per-device-train-batch-size 8
    --warmup-ratio 0.05
    --max-grad-norm 1.0
    --split-ratio 0.02
    --save-steps 400
    --eval-steps 200
    --save-total-limit 2
)

_run_one() {
    local out_dir="$1"
    local img_subdir="$2"
    echo "========================================================"
    echo "Running: output=${out_dir}, image-subdir=${img_subdir}"
    echo "========================================================"
    if [ "${N_GPUS}" -eq 1 ]; then
        python train_pdfa_colpali.py "${_COMMON_ARGS[@]}" --output-dir "${out_dir}" --image-subdir "${img_subdir}"
    else
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
        torchrun \
            --nproc_per_node="${N_GPUS}" \
            --master_port="${MASTER_PORT}" \
            train_pdfa_colpali.py "${_COMMON_ARGS[@]}" --output-dir "${out_dir}" --image-subdir "${img_subdir}" --dataloader-num-workers 0
    fi
    echo ""
}

_run_one "${OUT_BASE}/colqwen_pdfa_all_clip4" "pos_target_for_deepseek_images_clip4"

echo "All done. → ${OUT_BASE}/colqwen_pdfa_all_clip4"
