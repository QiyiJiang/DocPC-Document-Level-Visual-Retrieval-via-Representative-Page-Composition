#!/usr/bin/env python3
"""
读取 JSON，依次处理：
1. 只保留 query 在整份数据中仅出现 1 次的记录（重复的 query 整条删除）；
2. 对每条保留记录：pos_target_for_deepseek 只保留在 pos_target 中出现的项；
3. 对 pos_target_for_deepseek 去重并保持首次出现顺序。
结果保存到新文件。
"""
import json
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("/data/docpc_project/dataset_generate_new/pdfa/education/query_list_text_education_with_pos_target_for_deepseek.json")
OUTPUT_PATH = INPUT_PATH.parent / (INPUT_PATH.stem + "_cleaned.json")


def dedup_preserve_order(lst: list) -> list:
    """去重并保持首次出现顺序。"""
    return list(dict.fromkeys(lst))


def main():
    print(f"读取: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    # 1. 统计 query 出现次数，只保留 query 仅出现 1 次的记录
    n_original = len(data)
    query_count = Counter(rec.get("query", "") for rec in data)
    data = [rec for rec in data if query_count[rec.get("query", "")] == 1]
    n_removed_dup_query = n_original - len(data)
    print(f"原记录数: {n_original}，删除「query 重复」的记录后: {len(data)}（移除 {n_removed_dup_query} 条）")

    n_filtered = 0
    n_deduped = 0
    for rec in data:
        pos_target = rec.get("pos_target") or []
        pos_target_set = set(pos_target) if isinstance(pos_target, list) else set()
        qfd = rec.get("pos_target_for_deepseek") or []
        if not isinstance(qfd, list):
            rec["pos_target_for_deepseek"] = []
            continue

        # 2. 只保留在 pos_target 中出现的项
        filtered = [x for x in qfd if x in pos_target_set]
        if len(filtered) != len(qfd):
            n_filtered += 1

        # 3. 去重并保持顺序
        deduped = dedup_preserve_order(filtered)
        if len(deduped) != len(filtered):
            n_deduped += 1

        rec["pos_target_for_deepseek"] = deduped

    print(f"总记录数（输出）: {len(data)}")
    print(f"有从 pos_target_for_deepseek 中删除不在 pos_target 的项的记录数: {n_filtered}")
    print(f"去重后列表长度发生变化的记录数: {n_deduped}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
