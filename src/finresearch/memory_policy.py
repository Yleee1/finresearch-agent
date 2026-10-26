from __future__ import annotations

import re

_FACT_PATTERNS = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|usd|rmb|cny|million|billion)\b", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:元|万元|亿元|百万|千万|亿)"),
)
_FORBIDDEN = ("frozen", "gold evidence", "gold answer", "评分器", "冻结集答案")
_INSTRUCTION = re.compile(r"(?:ignore (?:all |previous )?instructions|system prompt|忽略.{0,8}指令|输出.{0,8}密钥)", re.IGNORECASE)


class MemoryPolicyError(ValueError):
    pass


def validate_procedural_content(content: str) -> str:
    value = " ".join(content.split()).strip()
    if len(value) < 8:
        raise MemoryPolicyError("memory content is too short to be a reusable strategy")
    lowered = value.lower()
    if any(token in lowered for token in _FORBIDDEN):
        raise MemoryPolicyError("frozen answers and scorer details cannot enter memory")
    if _INSTRUCTION.search(value):
        raise MemoryPolicyError("instruction-like document content cannot enter memory")
    if any(pattern.search(value) for pattern in _FACT_PATTERNS):
        raise MemoryPolicyError("unscoped financial values cannot enter procedural memory")
    return value
