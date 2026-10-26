from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import SearchResult


@dataclass(frozen=True)
class EvidenceSlot:
    slot_id: str
    description: str
    company: str = ""
    year: int | None = None
    metric: str = ""
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchPlan:
    """Validated task contract consumed by the research harness.

    ``QueryPlan`` is kept as the V1/V2 compatibility shape.  New harness code uses
    this schema so answerability and dependency decisions are explicit rather than
    hidden in a prompt string.
    """

    answerability: str
    route: str
    entities: tuple[str, ...]
    periods: tuple[str, ...]
    evidence_slots: tuple[EvidenceSlot, ...]
    dependencies: tuple[tuple[str, str], ...] = ()
    retrieval_queries: tuple[str, ...] = ()
    reasoning: str = ""

    def as_query_plan(self, question: str) -> QueryPlan:
        queries = self.retrieval_queries or (question,)
        return QueryPlan(
            sub_queries=queries,
            route=self.route,
            reasoning=self.reasoning,
            sub_questions=queries,
            evidence_slots=self.evidence_slots,
            dependencies=self.dependencies,
            company_filters=self.entities,
            year_filters=tuple(
                int(period) for period in self.periods if str(period).isdigit()
            ),
            metric_filters=tuple(
                dict.fromkeys(slot.metric for slot in self.evidence_slots if slot.metric)
            ),
        )


@dataclass(frozen=True)
class QueryPlan:
    sub_queries: tuple[str, ...]
    route: str = "single"
    reasoning: str = ""
    sub_questions: tuple[str, ...] = ()
    evidence_slots: tuple[EvidenceSlot, ...] = ()
    dependencies: tuple[tuple[str, str], ...] = ()
    company_filters: tuple[str, ...] = ()
    year_filters: tuple[int, ...] = ()
    metric_filters: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationDecision:
    sufficient: bool
    missing_evidence: tuple[str, ...] = ()
    follow_up_queries: tuple[str, ...] = ()
    reasoning: str = ""
    slot_coverage: tuple[tuple[str, str], ...] = ()
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    doc_id: str
    quote: str
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    normalized_subject: str = ""
    year: int | None = None
    metric: str = ""
    value: str = ""
    unit: str = ""
    citation_ids: tuple[str, ...] = ()
    supported: bool = False
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: tuple[Citation, ...]
    claims: tuple[Claim, ...] = ()


@dataclass(frozen=True)
class AnswerValidation:
    valid: bool
    issues: tuple[str, ...] = ()
    unsupported_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentTrace:
    iteration: int
    sub_queries: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    evidence_sufficient: bool
    missing_evidence: tuple[str, ...] = ()
    verification_reasoning: str = ""
    slot_coverage: tuple[tuple[str, str], ...] = ()
    follow_up_queries: tuple[str, ...] = ()
    newly_added_evidence: int = 0
    estimated_context_tokens: int = 0
    retrieval_seconds: float = 0.0
    verifier_seconds: float = 0.0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retry_gate_decision: str = ""
    retry_stop_reason: str = ""
    memory_hit_ids: tuple[str, ...] = ()


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
    claim_ledger: tuple[Claim, ...] = field(default_factory=tuple)
    llm_calls: int = 0
    end_to_end_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    harness_version: str = "v2-compat"
    memory_hit_ids: tuple[str, ...] = field(default_factory=tuple)
    stop_reason: str = ""
