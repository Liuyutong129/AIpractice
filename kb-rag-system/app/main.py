from fastapi import FastAPI
from pydantic import BaseModel
import os
import jieba
from dotenv import load_dotenv
from openai import OpenAI
app = FastAPI(title="KB RAG System")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZAI_API_KEY"),
    base_url=os.getenv("ZAI_BASE_URL")
)

MODEL_NAME = os.getenv("ZAI_MODEL", "glm-4.7-flash")

class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "hello, project is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


def split_text(text, max_sentences_per_chunk=2):
    # 先按中文句号切分
    sentences = []
    parts = text.replace("。", "。\n").splitlines()

    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)

        if len(current_chunk) >= max_sentences_per_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


def load_doc_chunks(folder="data/docs"):
    all_chunks = []

    if not os.path.exists(folder):
        return all_chunks

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            chunks = split_text(content, max_sentences_per_chunk=2)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "title": filename.replace(".txt", ""),
                    "chunk_id": i,
                    "content": chunk,
                    "source": path
                })

    return all_chunks


def search_chunks(question: str, top_k=2):
    chunks = load_doc_chunks()
    scored_chunks = []

    # 通用停用词
    stopwords = {
        "什么", "怎么", "怎么办", "一下", "一下子", "了", "的", "和", "与", "是"
    }

    # 领域里太泛的词，也先排掉
    weak_words = {
        "设备", "故障", "问题", "告警", "处理", "情况", "异常"
    }

    question_words = [
        w for w in jieba.lcut(question)
        if w.strip() and len(w) > 1 and w not in stopwords and w not in weak_words
    ]

    for chunk in chunks:
        score = 0
        matched_words = []

        for word in question_words:
            if word in chunk["content"]:
                score += 2
                matched_words.append(word)

        # 去重
        matched_words = list(dict.fromkeys(matched_words))

        # 至少命中 2 个有效词，才认为相关
        if len(matched_words) >= 2:
            scored_chunks.append({
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "source": chunk["source"],
                "score": score,
                "matched_words": matched_words
            })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]

def generate_answer(question: str, retrieved_chunks: list):
    if not retrieved_chunks:
        return "未找到相关信息。"

    context_text = "\n\n".join(
        [
            f"[片段{i+1}] 标题: {chunk['title']}\n内容: {chunk['content']}"
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
你是一个工业设备知识库问答助手。
请严格根据下面提供的检索片段回答问题，不要编造。
如果片段中没有足够信息，就明确回答“根据当前知识库内容，暂时无法确定”。

用户问题：
{question}

检索到的知识片段：
{context_text}

请用简洁、自然的中文回答，并在回答最后用“来源：...”标注你主要依据的片段标题。
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的工业知识库问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        answer = answer.replace("\n\n", " ")
        return answer
        
    except Exception as e:
        return f"大模型生成失败：{str(e)}"

@app.post("/query")
def query(request: QueryRequest):
    results = search_chunks(request.question, top_k=2)

    if results:
        answer = generate_answer(request.question, results)
        return {
            "question": request.question,
            "answer": answer,
            "retrieved_chunks": results,
            "status": "success"
        }

    return {
        "question": request.question,
        "answer": "未找到相关信息。",
        "retrieved_chunks": [],
        "status": "not_found"
    }