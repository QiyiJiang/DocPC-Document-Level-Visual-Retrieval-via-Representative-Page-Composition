#!/usr/bin/env bash
# Page-level PDFA 训练：先 ColQwen2.5 → 再 ColPali，线性串行
# 数据：expanded_x4 JSON（每条 1 query + 1 page 图片），loss = pairwise
#
# nohup 运行:
#   cd /data/docpc_project && nohup bash train_pdfa_page.sh > train_pdfa_page.log 2>&1 &
# 查看日志: tail -f train_pdfa_page.log

set -e
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

# ===== GPU 设置 =====
N_GPUS=1
CUDA_DEVICES="1"
MASTER_PORT=29504
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# ===== 数据路径 =====
JSON_ROOT="/data/docpc_project/dataset_generate_new/pdfa"
IMAGE_BASE="/data/docpc_project/datasets/pdfa"
IMAGE_SUFFIX="pos_target_for_deepseek_images_first4"
OUT_BASE="/data/docpc_project/models"

# ===== 公共训练参数 =====
COMMON_ARGS=(
    --json-root "${JSON_ROOT}"
    --image-base "${IMAGE_BASE}"
    --image-suffix "${IMAGE_SUFFIX}"
    --lr 5e-5
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

# ======================================================================
# Step 1: ColQwen2.5
# ======================================================================
echo "========================================================"
echo "Step 1/2: ColQwen2.5 page-level training"
echo "========================================================"

_COLQWEN_ARGS=(
    --model-type colqwen
    --pretrained-model-name-or-path /data/docpc_project/models/colqwen2.5_v0.2_merged
    --output-dir "${OUT_BASE}/colqwen_page_first4"
    "${COMMON_ARGS[@]}"
)

if [ "${N_GPUS}" -eq 1 ]; then
    python train_pdfa_page.py "${_COLQWEN_ARGS[@]}"
else
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
    torchrun \
        --nproc_per_node="${N_GPUS}" \
        --master_port="${MASTER_PORT}" \
        train_pdfa_page.py "${_COLQWEN_ARGS[@]}" --dataloader-num-workers 0
fi

echo ""
echo "ColQwen2.5 done."
echo ""

# ======================================================================
# Step 2: ColPali
# ======================================================================
echo "========================================================"
echo "Step 2/2: ColPali page-level training"
echo "========================================================"

_COLPALI_ARGS=(
    --model-type colpali
    --pretrained-model-name-or-path /data/docpc_project/models/colpali_v1.3_merged
    --output-dir "${OUT_BASE}/colpali_page_first4"
    "${COMMON_ARGS[@]}"
)

if [ "${N_GPUS}" -eq 1 ]; then
    python train_pdfa_page.py "${_COLPALI_ARGS[@]}"
else
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
    torchrun \
        --nproc_per_node="${N_GPUS}" \
        --master_port="${MASTER_PORT}" \
        train_pdfa_page.py "${_COLPALI_ARGS[@]}" --dataloader-num-workers 0
fi

echo ""
echo "All done! Both models trained and merged."
echo "  ColQwen merged → ${OUT_BASE}/colqwen_page_first4/sft-colqwen-merged"
echo "  ColPali merged → ${OUT_BASE}/colpali_page_first4/sft-colpali-merged"
