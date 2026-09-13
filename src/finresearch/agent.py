from __future__ import annotations

import re
from typing import Protocol

from .agent_schemas import (
    AnswerValidation,
    AgentTrace,
    GroundedAnswer,
    QueryPlan,
    ResearchResult,
    VerificationDecision,
)
from .schemas import SearchResult


class ResearchLLM(Protocol):
    def plan(self, question: str) -> QueryPlan: ...

    def verify(
        self, question: str, evidence: list[SearchResult]
    ) -> VerificationDecision: ...

    def answer(self, question: str, evidence: list[SearchResult]) -> GroundedAnswer: ...

    def validate_answer(
        self,
        question: str,
        answer: GroundedAnswer,
        evidence: list[SearchResult],
    ) -> AnswerValidation: ...


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", "", text)


def validate_citations(
    answer: GroundedAnswer, evidence: list[SearchResult]
) -> GroundedAnswer:
    """Only retain citations whose document and quote exist in retrieved evidence."""

    evidence_by_id = {item.doc_id: item for item in evidence}
    valid = []
    for citation in answer.citations:
        source = evidence_by_id.get(citation.doc_id)
        quote = _normalize_quote(citation.quote)
        if not source or not quote:
            continue
        if quote in _normalize_quote(source.text):
            valid.append(citation)
    return GroundedAnswer(answer=answer.answer, citations=tuple(valid))


class ResearchAgent:
    """Bounded retrieval agent with explicit state and a hard iteration limit."""

    def __init__(
        self,
        retriever,
        llm: ResearchLLM,
        *,
        top_k: int = 5,
        max_iterations: int = 2,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_iterations = max_iterations

    def _retrieve(
        self,
        queries: tuple[str, ...],
        accumulated: dict[str, SearchResult],
    ) -> None:
        for query in queries:
            for result in self.retriever.search(query, top_k=self.top_k):
                existing = accumulated.get(result.doc_id)
                if existing is None or result.score > existing.score:
                    accumulated[result.doc_id] = result

    def run(self, question: str) -> ResearchResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        plan = self.llm.plan(question)
        queries = plan.sub_queries or (question,)
        accumulated: dict[str, SearchResult] = {}
        traces: list[AgentTrace] = []

        for iteration in range(1, self.max_iterations + 1):
            self._retrieve(queries, accumulated)
            evidence = sorted(
                accumulated.values(),
                key=lambda item: (-item.score, item.doc_id),
            )
            decision = self.llm.verify(question, evidence)
            traces.append(
                AgentTrace(
                    iteration=iteration,
                    sub_queries=queries,
                    retrieved_doc_ids=tuple(item.doc_id for item in evidence),
                    evidence_sufficient=decision.sufficient,
                    missing_evidence=decision.missing_evidence,
                    verification_reasoning=decision.reasoning,
                )
            )

            if decision.sufficient:
                grounded = validate_citations(self.llm.answer(question, evidence), evidence)
                if grounded.citations:
                    validation = self.llm.validate_answer(question, grounded, evidence)
                    if not validation.valid:
                        return ResearchResult(
                            question=question,
                            route=plan.route,
                            status="abstained",
                            answer="答案语义校验失败，无法生成可靠答案。",
                            citations=(),
                            evidence=tuple(evidence),
                            iterations=iteration,
                            traces=tuple(traces),
                            refusal_reason="answer_grounding_failed: "
                            + "；".join(validation.issues),
                        )
                    return ResearchResult(
                        question=question,
                        route=plan.route,
                        status="answered",
                        answer=grounded.answer,
                        citations=grounded.citations,
                        evidence=tuple(evidence),
                        iterations=iteration,
                        traces=tuple(traces),
                    )
                return ResearchResult(
                    question=question,
                    route=plan.route,
                    status="abstained",
                    answer="证据引用校验失败，无法生成可靠答案。",
                    citations=(),
                    evidence=tuple(evidence),
                    iterations=iteration,
                    traces=tuple(traces),
                    refusal_reason="citation_validation_failed",
                )

            if iteration < self.max_iterations:
                queries = decision.follow_up_queries or decision.missing_evidence
                if not queries:
                    queries = (question,)

        evidence = sorted(
            accumulated.values(), key=lambda item: (-item.score, item.doc_id)
        )
        missing = traces[-1].missing_evidence if traces else ()
        detail = "、".join(missing) if missing else "回答所需的完整证据"
        return ResearchResult(
            question=question,
            route=plan.route,
            status="abstained",
            answer=f"现有资料不足，无法可靠回答；仍缺少：{detail}。",
            citations=(),
            evidence=tuple(evidence),
            iterations=self.max_iterations,
            traces=tuple(traces),
            refusal_reason="insufficient_evidence",
        )
