"""
基于文本相似组（similar_text_groups.json）和聚类关键词，用 DeepSeek 为每组生成一条检索 query。
组内的 text_names 即为正例（pos_target）；构造 query 时会附带候选正例的部分原文（与 keywords_deepseek_text 相同采样方式）。
"""
import asyncio
import os
import json
import random
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

# ============ 全部参数（在此修改） ============
# 数据目录；按类别使用 pdfa/<category>/ 下的 similar_text_groups_*.json 与 keywords_clusters_*.json
DATA_DIR = Path("/data/docpc_project/dataset_generate_new")
CATEGORY = "government"   # 类别名：biology / education / finance / government / industrial / legal / research
PDFA_DIR = DATA_DIR / "pdfa_test" / CATEGORY
SIMILAR_GROUPS_PATH = PDFA_DIR / f"similar_text_groups_{CATEGORY}.json"
KEYWORDS_CLUSTERS_PATH = PDFA_DIR / f"keywords_clusters_{CATEGORY}.json"
SAVE_PATH = PDFA_DIR / f"query_list_text_{CATEGORY}.json"   # 输出为单个 JSON，外层为 list

# 候选正例原文所在目录（与 keywords_deepseek_text 一致：按类别子目录）
TXT_BASE_DIR = Path("/data/docpc_project/datasets/pdfa_test_text_pymupdf")
TXT_DIR = TXT_BASE_DIR / CATEGORY

# 部分原文采样：与 keywords_deepseek_text.py 一致（前 2000 + 中间 8×500 + 后 2000，共 8000 字符）
HEAD_CHARS = 2000
TAIL_CHARS = 2000
MIDDLE_CHARS = 4000
MIDDLE_BLOCK_SIZE = 500
MIDDLE_BLOCK_COUNT = 8
MAX_EXCERPT_DOCS = 5   # 最多附带几篇文档的部分原文，避免 prompt 过长

BATCH_SIZE = 20          # 每批并发请求数
MAX_SAMPLES = None        # 最多处理多少组，None 表示全部
RANDOM_SEED = 42

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_MODEL = "deepseek-v3-2-251201"
# =============================================


def sample_text_head_middle_tail(text: str) -> str:
    """与 keywords_deepseek_text.py 一致：前 2000 + 中间 8×500 + 后 2000，共 8000 字符。"""
    total = HEAD_CHARS + MIDDLE_CHARS + TAIL_CHARS
    if len(text) <= total:
        return text
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]
    middle = text[HEAD_CHARS:-TAIL_CHARS]
    L = len(middle)
    if L <= MIDDLE_CHARS:
        middle_sampled = middle
    else:
        blocks = []
        for i in range(MIDDLE_BLOCK_COUNT):
            start = i * (L - MIDDLE_BLOCK_SIZE) // max(MIDDLE_BLOCK_COUNT - 1, 1)
            blocks.append(middle[start : start + MIDDLE_BLOCK_SIZE])
        middle_sampled = "".join(blocks)
    return head + middle_sampled + tail


def load_partial_texts(text_names: list[str], txt_dir: Path, max_docs: int = MAX_EXCERPT_DOCS) -> list[tuple[str, str]]:
    """按 text_names 从 txt_dir 读取文件，对每篇做 sample_text_head_middle_tail，返回 [(text_name, partial_text), ...]。最多 max_docs 篇。"""
    out = []
    for name in text_names[:max_docs]:
        path = txt_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue
        out.append((name, sample_text_head_middle_tail(text)))
    return out


QUERY_PROMPT = """
You will receive a group of similar **texts** (documents/papers) described by a shared keyword list and optional theme. When provided, you will also see **partial original content** (excerpts) of some of these documents (head + middle + tail sampling). These texts are grouped because they share topics; the keywords may be incomplete or inaccurate.

Your task is to imagine how a real user might describe what they are looking for. Based on the shared content of these texts (and the excerpts when given), generate a simple, natural question that such a user might type into a search engine — something they would actually say in daily life.

**Make it sound like a typical search. Avoid technical phrasing, abstract concepts, or long constraints.**

Guidelines:
1. The question should sound casual and realistic, as if the user is trying to find documents about a specific topic in a natural way.
2. Focus on the main shared theme among the texts. Do not try to cover every detail.
3. Only return an empty string if there's no meaningful connection and a realistic query cannot be formed.
4. Use the keywords only to support your understanding — do not treat them as guaranteed ground truth.
5. Prefer English for the query when the keywords are in English; use Chinese when the topic is better expressed in Chinese.

**Output format** (strictly follow):

<answer>
{
  "query": "your generated question here"
}
</answer>

If no reasonable query can be formed:

<answer>
{
  "query": ""
}
</answer>

Example query style:
- Find documents about Kerr solution and gravitational waves.
- Papers on gamma-ray bursts and Swift observations.
- Look for texts on quantum transport and Kondo regime.
"""


class DeepSeekQueryClient:
    def __init__(self):
        self.model = DEEPSEEK_MODEL
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.prompt = QUERY_PROMPT

    def _user_content(
        self,
        keyword_list: list,
        text_names: list,
        group_theme: str = "",
        text_excerpts: list[tuple[str, str]] | None = None,
    ) -> str:
        parts = [f"Shared keywords (by cluster): {json.dumps(keyword_list, ensure_ascii=False)}"]
        if group_theme and group_theme.strip():
            parts.append(f"Group theme (optional): {group_theme.strip()}")
        if text_excerpts:
            parts.append("\n--- Partial original content of some documents (for reference) ---")
            for i, (_, excerpt) in enumerate(text_excerpts, 1):
                parts.append(f"\n--- Excerpt {i} ---\n{excerpt}")
        return "\n".join(parts)

    async def generate_query(
        self,
        keyword_list: list,
        text_names: list,
        group_theme: str = "",
        text_excerpts: list[tuple[str, str]] | None = None,
    ) -> dict:
        """返回 {"query": "..."}；text_excerpts 为 [(text_name, partial_text), ...]，与 keywords_deepseek_text 相同采样。"""
        user = self._user_content(keyword_list, text_names, group_theme, text_excerpts)
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user},
        ]
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                stream=False,
            )
            raw = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  DeepSeek 调用失败: {e}")
            return {"query": ""}

        cleaned = raw.replace("<answer>", "").replace("</answer>", "").strip()
        try:
            out = json.loads(cleaned)
            return out if isinstance(out, dict) and "query" in out else {"query": ""}
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    out = json.loads(cleaned[start:end])
                    return out if isinstance(out, dict) and "query" in out else {"query": ""}
                except json.JSONDecodeError:
                    pass
        return {"query": ""}


async def query_create_one(
    keyword_list: list,
    text_names: list,
    group_theme: str,
    client: DeepSeekQueryClient,
    text_excerpts: list[tuple[str, str]] | None = None,
) -> dict:
    """单组生成 query，返回 {"query": ...}；可选传入 text_excerpts 作为部分原文。"""
    return await client.generate_query(keyword_list, text_names, group_theme, text_excerpts)


async def main():
    if SAVE_PATH.exists():
        print(f"{SAVE_PATH} 已存在，跳过")
        return

    print("Loading data...")
    with open(SIMILAR_GROUPS_PATH, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    with open(KEYWORDS_CLUSTERS_PATH, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    # cluster_id -> { "keywords": [...] }，与 query_create.py 中 keywords_dict 用法一致
    cid_to_info = {c["cluster_id"]: c for c in clusters}

    # 只使用 cluster_ids 数量 >= 5 的组
    data_list = [item for item in data_list if len(item.get("cluster_ids", [])) >= 5]
    if MAX_SAMPLES is not None and len(data_list) > MAX_SAMPLES:
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        data_list = random.sample(data_list, MAX_SAMPLES)
    print(f"Processing {len(data_list)} similar text groups (cluster_ids >= 5)...")

    client = DeepSeekQueryClient()
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_list = []   # 收集所有结果，最后统一写成外层 list

    for i in tqdm(range(0, len(data_list), BATCH_SIZE), desc="Batches"):
        batch = data_list[i : i + BATCH_SIZE]
        tasks = []
        for item in batch:
            cluster_ids = item["cluster_ids"]
            text_names = item["text_names"]
            keyword_list = [
                cid_to_info[cid]["keywords"]
                for cid in cluster_ids
                if cid in cid_to_info
            ]
            theme = item.get("group_theme") or ""
            # 加载候选正例的部分原文（与 keywords_deepseek_text 相同：前 2000 + 中间 8×500 + 后 2000）
            text_excerpts = load_partial_texts(text_names, TXT_DIR) if TXT_DIR.exists() else []
            tasks.append(
                query_create_one(keyword_list, text_names, theme, client, text_excerpts)
            )

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"Batch error: {e}")
            continue

        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"  Item error: {result}")
                continue
            if not result.get("query", "").strip():
                continue
            cluster_ids = item["cluster_ids"]
            keyword_list = [
                cid_to_info[cid]["keywords"]
                for cid in cluster_ids
                if cid in cid_to_info
            ]
            results_list.append({
                "query": result["query"].strip(),
                "pos_target": item["text_names"],
                "keyword_list": keyword_list,
                "cluster_ids": item["cluster_ids"],
            })

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote {len(results_list)} queries to {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
