#!/usr/bin/env python3
"""
统计 JSON 中 pos_target_for_deepseek 字段的不同大小及其对应数据量。
"""
import json
from pathlib import Path
from collections import Counter

# 可修改路径
JSON_PATH = Path("/data/docpc_project/dataset_generate_new/pdfa/education/query_list_text_education_with_pos_target_for_deepseek_cleaned.json")


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sizes = [len(rec.get("pos_target_for_deepseek") or []) for rec in data]
    counter = Counter(sizes)
    total = len(data)

    print(f"文件: {JSON_PATH.name}")
    print(f"总记录数: {total}")
    print()
    print("pos_target_for_deepseek 大小 -> 数据量")
    print("-" * 40)
    for size in sorted(counter.keys()):
        count = counter[size]
        pct = 100.0 * count / total
        print(f"  {size:6d}  ->  {count:6d}  ({pct:.1f}%)")
    print("-" * 40)
    print(f"校验: {sum(counter.values())} = {total}")


if __name__ == "__main__":
    main()
