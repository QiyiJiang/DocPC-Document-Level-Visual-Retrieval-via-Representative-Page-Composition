"""
按 similar_groups.py 的操作方式，对文本构建相似文本组：
1. 用 keywords_clusters.json 把 data_text_keywords.json 转成带 cluster_ids 的数据；
2. 按聚类 ID 组合找同时属于多个聚类的文本，形成相似文本组（每组 3~10 篇）；
3. 可选：调用 DeepSeek 为每组生成一句话主题标签。
"""
import asyncio
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from openai import AsyncOpenAI

# ============ 全部参数（在此修改） ============
SCRIPT_DIR = Path(__file__).resolve().parent
PDFA_DIR = SCRIPT_DIR / "pdfa_test"
KEYWORDS_DATA_PATH = PDFA_DIR / "biology" / "data_text_keywords_biology.json"   # 关键词 JSON
CLUSTER_PATH = PDFA_DIR / "biology" / "keywords_clusters_biology.json"   # 聚类结果保存路径
TEXT_WITH_CLUSTER_IDS_PATH = PDFA_DIR / "biology" /"data_text_with_biology_cluster_ids.json"  # 中间结果：带 cluster_ids 的文本
SIMILAR_GROUPS_PATH = PDFA_DIR /"biology" / "similar_text_groups_biology.json"     # 最终相似文本组

# 每组最少/最多文本数（对应原脚本的 3~5 张图）
MIN_TEXTS_PER_GROUP = 3
MAX_TEXTS_PER_GROUP = 5

# 聚类组合的 cluster_num：与 similar_groups 一致，可遍历 [2,3,...,10]
CLUSTER_NUMS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# 是否用 DeepSeek 为每组生成一句话主题
USE_DEEPSEEK_THEME = False
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_MODEL = "deepseek-v3-2-251201"

# 批量调用 DeepSeek 时每批组数（避免并发过高）
DEEPSEEK_BATCH_SIZE = 20
# =============================================

THEME_PROMPT = """你是一个主题概括助手。给定一组「相似文本」的聚类关键词和文本文件名列表，请用一句话概括该组的主题（英文或中文均可，简洁即可）。
只输出这一句话，不要解释、不要编号、不要引号。"""


def keywords_convert_to_cluster(data_path: Path, cluster_path: Path, save_path: Path) -> None:
    """将每条文本的关键词映射为 cluster_ids，写入 data_text_with_cluster_ids。"""
    with open(cluster_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    keyword_to_cluster = {}
    for cluster in clusters:
        cid = cluster["cluster_id"]
        for kw in cluster["keywords"]:
            keyword_to_cluster[kw] = cid

    with open(data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for item in data_list:
        if "keywords" in item:
            res = [keyword_to_cluster[kw] for kw in item["keywords"] if kw in keyword_to_cluster]
            item["cluster_ids"] = list(set(res))
        else:
            item["cluster_ids"] = []

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    print(f"已写入带 cluster_ids 的文本: {save_path}")


def load_cluster_id_to_keywords(cluster_path: Path) -> dict[int, list[str]]:
    """cluster_id -> 该聚类下的关键词列表，供 DeepSeek 主题生成用。"""
    with open(cluster_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    return {c["cluster_id"]: c["keywords"] for c in clusters}


def similar_groups_text(
    data_path: Path,
    save_path: Path,
    cluster_num: int,
    min_texts: int = MIN_TEXTS_PER_GROUP,
    max_texts: int = MAX_TEXTS_PER_GROUP,
) -> list[dict]:
    """
    与 similar_groups 相同逻辑，但用 text_name 替代 image_name。
    返回本次 cluster_num 下得到的相似文本组列表。
    """
    print(f"cluster_num: {cluster_num}")
    with open(data_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    inverted_index = defaultdict(set)
    for item in data_list:
        for cid in item.get("cluster_ids", []):
            inverted_index[cid].add(item["text_name"])

    visited_sets = set()
    results = []

    for item in data_list:
        cluster_ids = list(set(item.get("cluster_ids", [])))
        n_ids = len(cluster_ids)
        if n_ids == 0:
            continue

        combos = (
            [tuple(sorted(cluster_ids))]
            if n_ids < cluster_num
            else list(combinations(sorted(cluster_ids), cluster_num))
        )

        for combo in combos:
            texts = inverted_index[combo[0]].copy()
            for cid in combo[1:]:
                texts &= inverted_index[cid]

            if not (min_texts <= len(texts) <= max_texts):
                continue

            group_set = frozenset(texts)
            if group_set in visited_sets:
                continue

            visited_sets.add(group_set)
            results.append({
                "cluster_ids": list(combo),
                "text_names": sorted(group_set),
            })

    print(f"  cluster_num={cluster_num} 得到相似文本组: {len(results)} 组")
    return results


def merge_and_dedupe_groups(existing_path: Path, new_results: list[dict]) -> list[dict]:
    """与原有逻辑一致：若已有结果文件则读取并合并，按整条 item 去重。"""
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            before = json.load(f)
    else:
        before = []

    no_repeat = []
    for item in before + new_results:
        if item not in no_repeat:
            no_repeat.append(item)
    return no_repeat


# --------------- DeepSeek 为每组打主题 ---------------


class DeepSeekClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.model = model or DEEPSEEK_MODEL
        self.client = AsyncOpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )

    async def get_group_theme(
        self,
        cluster_ids: list[int],
        cid_to_keywords: dict[int, list[str]],
        text_names: list[str],
        max_tokens: int = 150,
    ) -> str:
        """根据聚类关键词和文本名列表，让模型返回一句话主题。"""
        keywords_parts = []
        for cid in cluster_ids:
            kws = cid_to_keywords.get(cid, [])
            keywords_parts.append(f"聚类 {cid}: " + ", ".join(kws[:15]))  # 每类最多 15 个词
        text_list = ", ".join(text_names[:20])  # 最多列 20 个文件名
        user_content = (
            "聚类关键词：\n" + "\n".join(keywords_parts) + "\n\n文本文件名：\n" + text_list
        )
        messages = [
            {"role": "system", "content": THEME_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
                stream=False,
            )
            out = (r.choices[0].message.content or "").strip()
            return out[:200] if out else ""
        except Exception as e:
            print(f"  DeepSeek 调用失败: {e}")
            return ""


async def add_themes_with_deepseek(
    groups: list[dict],
    cid_to_keywords: dict[int, list[str]],
) -> None:
    """就地为 groups 中每条添加 "group_theme" 字段。"""
    client = DeepSeekClient()
    for i in range(0, len(groups), DEEPSEEK_BATCH_SIZE):
        batch = groups[i : i + DEEPSEEK_BATCH_SIZE]
        tasks = [
            client.get_group_theme(
                g["cluster_ids"],
                cid_to_keywords,
                g["text_names"],
            )
            for g in batch
        ]
        themes = await asyncio.gather(*tasks, return_exceptions=True)
        for g, th in zip(batch, themes):
            g["group_theme"] = th if isinstance(th, str) else ""
        print(f"  已为 {min(i + DEEPSEEK_BATCH_SIZE, len(groups))}/{len(groups)} 组生成主题")


def main():
    # 1) 若尚未生成「带 cluster_ids 的文本」，则先转换
    if not TEXT_WITH_CLUSTER_IDS_PATH.exists():
        keywords_convert_to_cluster(
            data_path=KEYWORDS_DATA_PATH,
            cluster_path=CLUSTER_PATH,
            save_path=TEXT_WITH_CLUSTER_IDS_PATH,
        )
    else:
        print(f"已存在: {TEXT_WITH_CLUSTER_IDS_PATH}，跳过转换")

    # 2) 多组 cluster_num 跑相似文本组，并合并去重
    all_new_results = []
    for cn in CLUSTER_NUMS:
        part = similar_groups_text(
            data_path=TEXT_WITH_CLUSTER_IDS_PATH,
            save_path=SIMILAR_GROUPS_PATH,
            cluster_num=cn,
            min_texts=MIN_TEXTS_PER_GROUP,
            max_texts=MAX_TEXTS_PER_GROUP,
        )
        all_new_results.extend(part)

    final = merge_and_dedupe_groups(SIMILAR_GROUPS_PATH, all_new_results)
    print(f"合并去重后相似文本组总数: {len(final)}")

    # 3) 可选：用 DeepSeek 为每组生成一句话主题
    if USE_DEEPSEEK_THEME and final:
        cid_to_keywords = load_cluster_id_to_keywords(CLUSTER_PATH)
        asyncio.run(add_themes_with_deepseek(final, cid_to_keywords))

    SIMILAR_GROUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SIMILAR_GROUPS_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"已保存: {SIMILAR_GROUPS_PATH}")


if __name__ == "__main__":
    main()
