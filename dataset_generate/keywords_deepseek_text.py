"""
批量调用 DeepSeek 对文本生成关键词，并保存为 JSON。
输入：目录下的文本文件 或 含文本列表的 JSON；输出：带关键词的 JSON 文件。
Prompt 结构参考图片关键词提取任务，改为面向文本。
"""

import os
import json
import asyncio
from openai import AsyncOpenAI

# ==================== 配置 ====================

# 输入：文本文件所在目录（每个 .txt 文件为一条文本）
TEXT_DIR = "/data/docpc_project/datasets/pdfa_test_text_pymupdf"
# 或：从 JSON 读取文本列表时使用，格式 [{"id": "1", "text": "..."}, ...]
TEXT_JSON_PATH = None  # 例如 "gold_datasets/texts.json"

# 输出
SAVE_PATH = "/data/docpc_project/dataset_generate_new/pdfa_test/data_text_keywords.json"

# 批量大小
BATCH_SIZE = 50

# 单条文本采样总长 8000 字符：前 2000 + 中间均匀 4000 + 后 2000（见 sample_text_head_middle_tail）

# DeepSeek 客户端配置（与你的调用方式一致）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEEPSEEK_MODEL = "deepseek-v3-2-251201"

# ==================== 文本关键词 + 类别 Prompt ====================

KEYWORDS_PROMPT = """
# Role Description
- You are a language model specialized in extracting keywords and classifying document type from text content, for use in indexing and retrieval.

# Responsibilities
1. Extract 3 to 20 concise, informative keywords from the input text.
2. Assign exactly one document category that best describes the text type/subject.

# Keyword Extraction Rules (Focus: Objectivity + Retrieval-Oriented)
- The language of the keywords must match the language used in the text — Chinese for Chinese text, English for English content.
- Duplicate keywords are strictly prohibited.
- Keywords should reflect main topics, entities, actions, events, and important phrases in the text.
- Do NOT include meaningless tokens: single letters, standalone symbols, numbers without context, or overly generic words.
- Keywords must be semantically meaningful and grounded in the text. Always provide a keyword list.

# Document Category
- Choose ONE category that best fits the document. Prefer the following or similar standard types (output in Chinese or English consistently):
  - 能源报告 / energy report
  - 政府报告 / government report
  - AI科研 / AI research
  - 物理科研 / physics research
  - 计算机/软件 / computer science, software
  - 生物/医药 / biology, pharmaceutical
  - 金融/经济 / finance, economics
  - 法律 / legal
  - 教育 / education
  - 工业/制造 / industrial, manufacturing
  - 其他 / other (use only when none of the above fit; you may give a short label like "技术白皮书")
- The category should objectively reflect the main subject or document type (e.g., academic paper, report, standard, manual).

# Output Format
The output must be strictly limited to the following JSON format, with no additional text:
<answer>
{
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "category": "类别名称，如：AI科研 或 energy report"
}
</answer>
"""


# ==================== DeepSeek 客户端 ====================


class DeepSeekClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.model = model or DEEPSEEK_MODEL
        self.client = AsyncOpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.keywords_prompt = KEYWORDS_PROMPT

    async def get_keywords(self, text: str, temperature=0.3, max_tokens=1024):
        """对单条文本提取关键词和类别，返回 {"keywords": [...], "category": "..."}。"""
        messages = [
            {"role": "system", "content": self.keywords_prompt},
            {"role": "user", "content": text},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            response_text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"API 调用失败: {e}")
            return {"keywords": [], "category": ""}

        cleaned = response_text.replace("<answer>", "").replace("</answer>", "").strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    obj = json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    obj = {}
            else:
                obj = {}
        return {
            "keywords": obj.get("keywords") or [],
            "category": (obj.get("category") or "").strip() or "",
        }


# ==================== 输入采样（前 2000 + 中间 8 块×500 + 后 2000，共 8000 字符） ====================

HEAD_CHARS = 2000
TAIL_CHARS = 2000
MIDDLE_CHARS = 4000  # 8 块 × 500
MIDDLE_BLOCK_SIZE = 500
MIDDLE_BLOCK_COUNT = 8  # 中间段分为 8 个块，每块 500 字符，均匀从中间文本中取
TOTAL_SAMPLE_CHARS = HEAD_CHARS + MIDDLE_CHARS + TAIL_CHARS  # 8000


def sample_text_head_middle_tail(text: str) -> str:
    """取前 2000、后 2000；中间 4000 字符 = 8 个 500 字符的块，在中间文本上均匀取这 8 块。不足 8000 则全文返回。"""
    if len(text) <= TOTAL_SAMPLE_CHARS:
        return text
    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]
    middle = text[HEAD_CHARS : -TAIL_CHARS]
    L = len(middle)
    if L <= MIDDLE_CHARS:
        middle_sampled = middle
    else:
        # 在中间段上均匀取 8 段连续的 500 字符块（每块连续，不逐字采样）
        # 8 块的起始位置：0, (L-500)/7, 2(L-500)/7, ..., 7(L-500)/7
        blocks = []
        for i in range(MIDDLE_BLOCK_COUNT):
            start = i * (L - MIDDLE_BLOCK_SIZE) // (MIDDLE_BLOCK_COUNT - 1)
            blocks.append(middle[start : start + MIDDLE_BLOCK_SIZE])
        middle_sampled = "".join(blocks)
    return head + middle_sampled + tail


# ==================== 数据加载 ====================


def load_texts_from_dir(dir_path: str):
    """从目录加载所有 .txt 文件，返回 [(name, text), ...]。"""
    if not os.path.isdir(dir_path):
        return []
    out = []
    for name in sorted(os.listdir(dir_path)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(dir_path, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"读取失败 {path}: {e}")
            continue
        out.append((name, text))
    return out


def load_texts_from_json(json_path: str):
    """从 JSON 加载文本列表。期望格式 [{"id": "1", "text": "..."}, ...] 或 [{"text": "..."}, ...]。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    out = []
    for i, item in enumerate(data):
        text = item.get("text") or item.get("content") or ""
        name = item.get("id") or item.get("name") or f"text_{i}"
        if isinstance(name, int):
            name = str(name)
        out.append((name, text))
    return out


# ==================== 单条与批量 ====================


async def keywords_one(client: DeepSeekClient, name: str, text: str):
    """对单条 (name, text) 提取关键词和类别，返回 {"text_name", "keywords", "category"} 或 None。"""
    if not text.strip():
        return None
    text = sample_text_head_middle_tail(text)
    result = await client.get_keywords(text)
    keywords = result.get("keywords") or []
    if not keywords:
        return None
    category = result.get("category") or ""
    row = {"text_name": name, "keywords": keywords, "category": category}
    print(row)
    return row


def load_existing_keywords(save_path: str):
    """若输出文件已存在，读取并返回 (已有记录列表, 已有 text_name 集合)。"""
    if not os.path.isfile(save_path):
        return [], set()
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return [], set()
        existing = list(data)
        names = {r.get("text_name") for r in existing if r.get("text_name") is not None}
        return existing, names
    except Exception as e:
        print(f"读取已有 JSON 失败 {save_path}: {e}，将重新生成")
        return [], set()


async def main():
    # 读取已有结果，跳过已存在 text_name 的条目
    existing_data, existing_names = load_existing_keywords(SAVE_PATH)
    if existing_names:
        print(f"已有 {len(existing_names)} 条记录，将跳过这些 text_name")

    # 选择输入源
    if TEXT_JSON_PATH and os.path.isfile(TEXT_JSON_PATH):
        items = load_texts_from_json(TEXT_JSON_PATH)
        print(f"从 JSON 加载 {len(items)} 条文本: {TEXT_JSON_PATH}")
    else:
        items = load_texts_from_dir(TEXT_DIR)
        print(f"从目录加载 {len(items)} 个文本文件: {TEXT_DIR}")

    # 只处理尚未有关键词的
    to_process = [(name, text) for name, text in items if name not in existing_names]
    skipped = len(items) - len(to_process)
    if skipped:
        print(f"跳过已存在 {skipped} 条，待处理 {len(to_process)} 条")
    if not to_process:
        print("没有需要新生成关键词的文本，直接保存已有结果")
        os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        print(f"共 {len(existing_data)} 条已保存到 {SAVE_PATH}")
        return

    client = DeepSeekClient()
    data_list = []
    total = len(to_process)

    for i in range(0, total, BATCH_SIZE):
        batch = to_process[i : i + BATCH_SIZE]
        tasks = [keywords_one(client, name, text) for name, text in batch]
        try:
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            for j, ans in enumerate(answers):
                if isinstance(ans, Exception):
                    print(f"处理失败 {batch[j][0]}: {ans}")
                elif ans is not None:
                    data_list.append(ans)
        except Exception as e:
            print(f"批次 {i} 错误: {e}")
        print(f"已处理 {min(i + BATCH_SIZE, total)}/{total} 条")

    # 合并已有 + 新生成的，写回
    final_list = existing_data + data_list
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)
    print(f"本次新增 {len(data_list)} 条，合计 {len(final_list)} 条已保存到 {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
