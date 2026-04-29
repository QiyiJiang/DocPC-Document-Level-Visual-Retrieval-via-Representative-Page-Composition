#!/usr/bin/env bash
# 按文档页面数量分组，基于已有全量检索结果分析召回效果
# 不需要 GPU，纯分析脚本
#
# 用法:
#   cd /data/docpc_project && bash eval_by_page_group.sh

set -e

# ==================== 在这里修改参数 ====================
# 已有检索结果目录（combo 测评结果所在目录）、

RETRIEVAL_DIR="/data/docpc_project/500——eval_results/combo"

# 测评结果文件名前缀，对应 eval_xxx__strategy__{cat}.json 里的 eval_xxx__strategy
# 例如: eval_colqwen_pdfa_all_first4__first4 → 加载 eval_colqwen_pdfa_all_first4__first4__biology.json 等
EVAL_NAME="eval_colqwen_pdfa_all_first4__first4"

# 输出目录（按页数分组的汇总结果）
OUTPUT_DIR="/data/docpc_project/500——eval_results/by_page_group"
# ======================================================

echo "=============================================="
echo "按页面数量分组分析检索效果"
echo "检索结果目录: ${RETRIEVAL_DIR}"
echo "测评文件前缀: ${EVAL_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=============================================="

python eval_by_page_group.py \
    --retrieval-dir "${RETRIEVAL_DIR}" \
    --eval-name "${EVAL_NAME}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "分析完成！结果在: ${OUTPUT_DIR}"
