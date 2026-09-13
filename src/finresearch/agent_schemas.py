from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import SearchResult


@dataclass(frozen=True)
class QueryPlan:
    sub_queries: tuple[str, ...]
    route: str = "single"
    reasoning: str = ""


@dataclass(frozen=True)
class VerificationDecision:
    sufficient: bool
    missing_evidence: tuple[str, ...] = ()
    follow_up_queries: tuple[str, ...] = ()
    reasoning: str = ""


@dataclass(frozen=True)
class Citation:
    doc_id: str
    quote: str


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class AnswerValidation:
    valid: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentTrace:
    iteration: int
    sub_queries: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    evidence_sufficient: bool
    missing_evidence: tuple[str, ...] = ()
    verification_reasoning: str = ""


@dataclass(frozen=True)
class ResearchResult:
    question: str
    route: str
    status: str
    answer: str
    citations: tuple[Citation, ...]
    evidence: tuple[SearchResult, ...]
    iterations: int
    traces: tuple[AgentTrace, ...] = field(default_factory=tuple)
    refusal_reason: str = ""
