import json
import os
import random

PDFA_ROOT = "/data/docpc_project/dataset_generate_new/pdfa"
CATEGORIES = ["biology", "education", "finance", "government", "industrial", "legal", "research"]
SEED = 42

for cat in CATEGORIES:
    src = os.path.join(PDFA_ROOT, cat, f"query_list_text_{cat}_with_pos_target_for_deepseek_cleaned.json")
    dst = os.path.join(PDFA_ROOT, cat, f"query_list_text_{cat}_with_pos_target_for_deepseek_cleaned_expanded_x4.json")

    if not os.path.exists(src):
        print(f"[{cat}] JSON not found, skip: {src}")
        continue

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(SEED)
    expanded = []
    for entry in data:
        query = entry["query"]
        targets = entry["pos_target_for_deepseek"]
        if not targets:
            continue
        target = targets[0]
        stem, ext = os.path.splitext(target)
        i = random.randint(0, 3)
        expanded.append({
            "query": query,
            "pos_target": f"{stem}_{i}{ext}"
        })

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)

    print(f"[{cat}] {len(data)} → {len(expanded)} (1:1), saved: {dst}")

print("Done.")
