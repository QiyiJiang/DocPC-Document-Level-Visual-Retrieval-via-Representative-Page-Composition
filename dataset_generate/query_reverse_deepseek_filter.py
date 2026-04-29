"""
基于 data_text_with_cluster_ids.json（文件–querys 一对多），对每条记录：
从 TEXT_DIR 按 text_name 读取完整文件内容（不截断），与该条的 querys 一起交给 DeepSeek，
判断其中哪些 query 与该文件相关，结果写入 querys_for_deepseek（列表类型）。

超过 MAX_DOC_CHARS 的文档会自动截断到该长度后再送入模型。
"""
import asyncio
import os
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

# ============ 全部参数（在此修改） ============
DATA_DIR = Path("/data/docpc_project/dataset_generate_new")
CATEGORY = "government"   # 类别名：biology / education / finance / government / industrial / legal / research
PDFA_DIR = DATA_DIR / "pdfa_test" / CATEGORY

DATA_TEXT_PATH = PDFA_DIR / f"data_text_with_{CATEGORY}_cluster_ids_with_querys.json"
TEXT_DIR = Path("/data/docpc_project/datasets/pdfa_test_text_pymupdf") / CATEGORY
OUT_PATH = PDFA_DIR / f"data_text_with_{CATEGORY}_cluster_ids_with_querys_for_deepseek.json"

# 并发数：同时进行中的 API 请求上限（仅对 querys 非空的记录发起请求）
CONCURRENCY = 3
# 每批条数（用于进度条，每批内最多 CONCURRENCY 个同时在请求）
BATCH_SIZE = 30

# 单文档最大字符数，超出则截断（预留 system + queries 空间，128k 上下文约 25 万字符内）
MAX_DOC_CHARS = 150000

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_MODEL = "deepseek-v3-2-251201"
# =============================================

SYSTEM_PROMPT = """You are a retrieval judge. You will be given:
1. One document (full text content).
2. A list of candidate search queries that were previously associated with this document.

Your task: Decide which of these queries are semantically relevant to the document. A query is relevant if a user searching with that query would reasonably expect this document in the results.

- Include a query if the document content matches the query's topic or intent, even partially.
- Exclude only queries that are clearly unrelated to the document.
- Return exactly the list of relevant query strings (as given in the input). No extra text.

Output format (strictly follow):
<answer>
{
  "querys_for_deepseek": ["query1", "query2"]
}
</answer>

If none are relevant:
<answer>
{
  "querys_for_deepseek": []
}
</answer>"""


def truncate_doc(content: str, max_chars: int, suffix: str = "…") -> str:
    """超过 max_chars 时截断，末尾加 suffix。"""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return content[: max_chars - len(suffix)] + suffix


def load_doc_full(text_dir: Path, text_name: str, max_chars: int | None = None) -> str:
    """读取文件内容；若指定 max_chars 且超出则截断。"""
    path = text_dir / text_name
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return ""
    if max_chars is not None and len(content) > max_chars:
        content = truncate_doc(content, max_chars)
    return content


class DeepSeekReverseClient:
    def __init__(self):
        self.model = DEEPSEEK_MODEL
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def _user_content(self, doc_content: str, querys: list[str]) -> str:
        lines = ["--- document content ---", doc_content, "", "--- candidate queries ---"]
        for i, q in enumerate(querys, 1):
            lines.append(f"{i}. {q}")
        return "\n".join(lines)

    async def filter_querys(
        self,
        doc_content: str,
        querys: list[str],
        valid_querys: set[str],
    ) -> list[str]:
        """给定文档全文和候选 query 列表，返回判定相关的 query 子集（必须在 valid_querys 内）。"""
        if not querys:
            return []
        user = self._user_content(doc_content, querys)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
                stream=False,
            )
            raw = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  DeepSeek 调用失败: {e}")
            return []

        cleaned = raw.replace("<answer>", "").replace("</answer>", "").strip()
        try:
            out = json.loads(cleaned)
            chosen = out.get("querys_for_deepseek") or []
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    out = json.loads(cleaned[start:end])
                    chosen = out.get("querys_for_deepseek") or []
                except json.JSONDecodeError:
                    chosen = []
            else:
                chosen = []
        return [q for q in chosen if q in valid_querys]


async def process_one(
    record: dict,
    text_dir: Path,
    client: DeepSeekReverseClient,
    sem: asyncio.Semaphore,
) -> dict:
    """对一条记录：读全文 + querys，调用 DeepSeek，写回 querys_for_deepseek。"""
    async with sem:
        record = dict(record)
        text_name = record.get("text_name") or ""
        querys = list(record.get("querys") or [])
        if not text_name:
            record["querys_for_deepseek"] = []
            return record
        if not querys:
            record["querys_for_deepseek"] = []
            return record

        doc_content = load_doc_full(text_dir, text_name, max_chars=MAX_DOC_CHARS)
        valid = set(querys)
        chosen = await client.filter_querys(doc_content, querys, valid)
        record["querys_for_deepseek"] = chosen
        return record


async def main():
    print("Loading data_text_with_cluster_ids.json...")
    with open(DATA_TEXT_PATH, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if not isinstance(data_list, list):
        data_list = [data_list]

    # 只对 querys 非空的记录请求 API，为空的直接置 querys_for_deepseek=[]
    need_judge = []  # (index, record)
    for i, rec in enumerate(data_list):
        querys = rec.get("querys") or []
        if querys and rec.get("text_name"):
            need_judge.append((i, rec))
        else:
            rec = dict(rec)
            rec["querys_for_deepseek"] = []
            data_list[i] = rec
    n_total = len(data_list)
    n_need = len(need_judge)
    print(f"共 {n_total} 条，其中 {n_need} 条 querys 非空需判断，其余 {n_total - n_need} 条直接置空。")

    text_dir = Path(TEXT_DIR)
    if not text_dir.is_dir():
        raise FileNotFoundError(f"文本目录不存在: {text_dir}")

    client = DeepSeekReverseClient()
    sem = asyncio.Semaphore(CONCURRENCY)

    # 只对 need_judge 并发：每批 BATCH_SIZE 条，批内最多 CONCURRENCY 个同时请求
    results_by_index = {}  # index -> result record
    num_batches = max(1, (n_need + BATCH_SIZE - 1) // BATCH_SIZE)
    for start in tqdm(range(0, n_need, BATCH_SIZE), desc="Reverse filter", total=num_batches):
        batch = need_judge[start : start + BATCH_SIZE]
        tasks = [process_one(rec, text_dir, client, sem) for _, rec in batch]
        results_ordered = await asyncio.gather(*tasks, return_exceptions=True)
        for (idx, _), r in zip(batch, results_ordered):
            if isinstance(r, Exception):
                print(f"Error index {idx}: {r!r}")
                rec = dict(data_list[idx])
                rec["querys_for_deepseek"] = list(rec.get("querys") or [])
                results_by_index[idx] = rec
            else:
                results_by_index[idx] = r

    for idx, rec in enumerate(data_list):
        if idx in results_by_index:
            data_list[idx] = results_by_index[idx]
    out_list = data_list

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote {len(out_list)} items to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
