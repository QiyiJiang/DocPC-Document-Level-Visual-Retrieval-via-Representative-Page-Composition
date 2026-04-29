"""
利用 data_text_with_querys_for_deepseek.json 中「文件 → querys_for_deepseek」的一对多关系，
建立「query → 作为该 query 正例的 text_name 列表」反向索引，
给 query_list_text.json 每条数据增加字段 pos_target_for_deepseek（DeepSeek 判定的该 query 的正例文件列表）。
"""
import json
from pathlib import Path

# ============ 路径 ============
DATA_DIR = Path("/data/docpc_project/dataset_generate_new")
CATEGORY = "government"   # 类别名：biology / education / finance / government / industrial / legal / research
PDFA_DIR = DATA_DIR / "pdfa_test" / CATEGORY
DATA_TEXT_PATH = PDFA_DIR / f"data_text_with_{CATEGORY}_cluster_ids_with_querys_for_deepseek.json"
QUERY_LIST_PATH = PDFA_DIR / f"query_list_text_{CATEGORY}.json"
OUT_PATH = PDFA_DIR / f"query_list_text_{CATEGORY}_with_pos_target_for_deepseek.json"  # 可改为 QUERY_LIST_PATH 则写回原文件
# =============================================


def main():
    print("Loading data_text_with_querys_for_deepseek.json...")
    with open(DATA_TEXT_PATH, "r", encoding="utf-8") as f:
        data_text_list = json.load(f)
    if not isinstance(data_text_list, list):
        data_text_list = [data_text_list]

    # query -> [text_name, ...]（该 query 被 DeepSeek 判定为正例的文件列表）
    query_to_files = {}
    for rec in data_text_list:
        text_name = rec.get("text_name")
        if not text_name:
            continue
        for q in rec.get("querys_for_deepseek") or []:
            q = (q or "").strip()
            if not q:
                continue
            if q not in query_to_files:
                query_to_files[q] = []
            query_to_files[q].append(text_name)

    print(f"Built index: {len(query_to_files)} queries with at least one DeepSeek positive file.")

    print("Loading query_list_text.json...")
    with open(QUERY_LIST_PATH, "r", encoding="utf-8") as f:
        query_list = json.load(f)
    if not isinstance(query_list, list):
        query_list = [query_list]

    for item in query_list:
        q = (item.get("query") or "").strip()
        item["pos_target_for_deepseek"] = list(query_to_files.get(q, []))

    print(f"Writing {len(query_list)} items to {OUT_PATH}...")
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(query_list, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
