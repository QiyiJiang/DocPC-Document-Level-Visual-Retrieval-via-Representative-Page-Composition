"""
根据 query_list_text.json 的 query–pos_target 关系，给 data_text_with_cluster_ids.json
的每条数据增加字段 querys：该 text_name 作为正例时对应的所有 query，类型为列表。
"""
import json
from pathlib import Path

# ============ 路径 ============ß
DATA_DIR = Path("/data/docpc_project/dataset_generate_new")
CATEGORY = "government"   # 类别名：biology / education / finance / government / industrial / legal / research
PDFA_DIR = DATA_DIR / "pdfa_test" / CATEGORY
QUERY_LIST_PATH = PDFA_DIR / f"query_list_text_{CATEGORY}.json"
DATA_TEXT_PATH = PDFA_DIR / f"data_text_with_{CATEGORY}_cluster_ids.json"
OUT_PATH = PDFA_DIR / f"data_text_with_{CATEGORY}_cluster_ids_with_querys.json"  # 写回同一文件，可改为新路径

# =============================================ß


def main():
    print("Loading query_list_text.json...")
    with open(QUERY_LIST_PATH, "r", encoding="utf-8") as f:
        query_list = json.load(f)
    if not isinstance(query_list, list):
        query_list = [query_list]

    # text_name -> [query, ...]（该文件作为正例的所有 query）
    name_to_querys = {}
    for item in query_list:
        q = (item.get("query") or "").strip()
        if not q:
            continue
        for name in item.get("pos_target") or []:
            if name not in name_to_querys:
                name_to_querys[name] = []
            name_to_querys[name].append(q)

    print(f"Loaded {len(query_list)} queries, {len(name_to_querys)} distinct text_name as positive.")

    print("Loading data_text_with_cluster_ids.json...")
    with open(DATA_TEXT_PATH, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if not isinstance(data_list, list):
        data_list = [data_list]

    for rec in data_list:
        name = rec.get("text_name")
        rec["querys"] = name_to_querys.get(name, [])

    print(f"Writing {len(data_list)} records to {OUT_PATH}...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
