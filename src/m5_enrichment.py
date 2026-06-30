from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def get_llm_client_and_model(default_model="gpt-4o-mini"):
    import os
    from openai import OpenAI
    
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    ds_base = os.getenv("DEEPSEEK_API_BASE")
    ds_model = os.getenv("DEEPSEEK_MODEL_NAME")
    
    if ds_key:
        client = OpenAI(api_key=ds_key, base_url=ds_base)
        model = ds_model or "deepseek-chat"
        return client, model

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY:
        client = OpenAI()
        return client, default_model
        
    return None, ""


def parse_json_robust(text: str) -> dict:
    import json as _json
    import re
    
    if not text:
        return {}
    
    text = text.strip()
    
    # Strip markdown code block wrappers
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    try:
        return _json.loads(text, strict=False)
    except Exception as e:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return _json.loads(text[start:end+1], strict=False)
        except Exception:
            pass
        raise e


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    client, model = get_llm_client_and_model("gpt-4o-mini")
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  Summarize failed: {e}")

    # Extractive fallback (không cần API):
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    client, model = get_llm_client_and_model("gpt-4o-mini")
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
        except Exception as e:
            print(f"  ⚠️  HyQA failed: {e}")

    # Extractive fallback:
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    client, model = get_llm_client_and_model("gpt-4o-mini")
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  Contextual failed: {e}")

    # Simple fallback:
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    client, model = get_llm_client_and_model("gpt-4o-mini")
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về dưới dạng json: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=250,
            )
            return parse_json_robust(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  Metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    client, model = get_llm_client_and_model("gpt-4o-mini")
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về dưới dạng json:
{
  "summary": "tóm tắt 2-3 câu bằng tiếng Việt",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=1000,
            )
            return parse_json_robust(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
