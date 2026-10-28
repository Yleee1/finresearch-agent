from __future__ import annotations

import re
from dataclasses import replace
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from .agent_schemas import (
    AgentTrace,
    AnswerValidation,
    GroundedAnswer,
    QueryPlan,
    ResearchResult,
    VerificationDecision,
)
from .answer_critic import AnswerCritic
from .budgets import ResearchBudget
from .evidence import EvidencePackBuilder, pack_to_search_results
from .planner import PlannerValidationError, StructuredPlanner
from .query_analysis import QueryAnalyzer
from .retrieval_critic import RetrievalCritic, merge_decisions
from .retrieval_value_gate import RetrievalValueGate
from .schemas import SearchResult
from .trace_events import NullTraceSink, TraceEvent


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
    return GroundedAnswer(
        answer=answer.answer,
        citations=tuple(valid),
        claims=answer.claims,
    )


def _find_documents(retriever) -> list:
    """Unwrap common retriever decorators to locate the immutable corpus."""

    documents = getattr(retriever, "documents", None)
    if isinstance(documents, list):
        return documents
    for attribute in ("base_retriever", "sparse_retriever"):
        nested = getattr(retriever, attribute, None)
        if nested is not None:
            found = _find_documents(nested)
            if found:
                return found
    return []


class ResearchAgent:
    """Bounded retrieval agent with explicit state and a hard iteration limit."""

    def __init__(
        self,
        retriever,
        llm: ResearchLLM,
        *,
        top_k: int = 5,
        max_iterations: int = 2,
        budget: ResearchBudget | None = None,
        expand_neighbors: bool = True,
        decompose_queries: bool = True,
        use_evidence_slots: bool = True,
        require_verification: bool = True,
        use_retrieval_critic: bool = True,
        allow_reflection: bool = True,
        use_answer_critic: bool = True,
        structured_planner: StructuredPlanner | None = None,
        retrieval_value_gate: RetrievalValueGate | None = None,
        procedural_memory=None,
        trace_sink=None,
        harness_version: str = "v3-dev-0",
        dataset_split: str = "interactive",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.budget = budget or ResearchBudget(max_retrieval_rounds=max_iterations)
        self.max_iterations = min(max_iterations, self.budget.max_retrieval_rounds)
        self.expand_neighbors = expand_neighbors
        self.decompose_queries = decompose_queries
        self.use_evidence_slots = use_evidence_slots
        self.require_verification = require_verification
        self.use_retrieval_critic = use_retrieval_critic
        self.allow_reflection = allow_reflection
        self.use_answer_critic = use_answer_critic
        self.pack_builder = EvidencePackBuilder(_find_documents(retriever))
        self.retrieval_critic = RetrievalCritic()
        self.answer_critic = AnswerCritic()
        self.structured_planner = structured_planner or StructuredPlanner(llm)
        self.retrieval_value_gate = retrieval_value_gate or RetrievalValueGate()
        self.procedural_memory = procedural_memory
        self.trace_sink = trace_sink or NullTraceSink()
        self.harness_version = harness_version
        self.dataset_split = dataset_split

    def _retrieve(
        self,
        queries: tuple[str, ...],
        accumulated: dict[str, SearchResult],
        provenance: list[tuple[SearchResult, str, int]],
        iteration: int,
    ) -> int:
        before = set(accumulated)
        for query in queries[: self.budget.max_sub_queries]:
            for result in self.retriever.search(query, top_k=self.top_k):
                provenance.append((result, query, iteration))
                existing = accumulated.get(result.doc_id)
                if existing is None or result.score > existing.score:
                    accumulated[result.doc_id] = result
        return len(set(accumulated) - before)

    @staticmethod
    def _budget_refusal(
        question: str,
        route: str,
        evidence: list[SearchResult],
        iterations: int,
        traces: list[AgentTrace],
        reason: str,
        llm_calls: int,
    ) -> ResearchResult:
        return ResearchResult(
            question=question,
            route=route,
            status="abstained",
            answer=f"已触及研究预算，系统安全拒答：{reason}。",
            citations=(),
            evidence=tuple(evidence),
            iterations=iterations,
            traces=tuple(traces),
            refusal_reason=f"budget_exceeded:{reason}",
            llm_calls=llm_calls,
        )

    def run(self, question: str) -> ResearchResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        started = perf_counter()
        run_id = f"run-{uuid4().hex}"
        trace_id = f"trace-{uuid4().hex}"
        client = getattr(self.llm, "client", None)
        initial_usage = dict(getattr(client, "usage", {}) or {})
        memory_hits = ()

        def emit(stage: str, *, refs=(), summary=None, latency_ms=0.0, error="") -> None:
            self.trace_sink.emit(TraceEvent(
                trace_id=trace_id,
                run_id=run_id,
                dataset_split=self.dataset_split,
                harness_version=self.harness_version,
                stage=stage,
                input_refs=tuple(str(item) for item in refs),
                output_summary=dict(summary or {}),
                latency_ms=latency_ms,
                token_usage={
                    field: max(usage_delta(field), 0)
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                model=str(getattr(client, "model", "")),
                error_code=error,
            ))

        def usage_delta(field: str) -> int:
            current = dict(getattr(client, "usage", {}) or {})
            return int(current.get(field, 0)) - int(initial_usage.get(field, 0))

        def finish(result: ResearchResult) -> ResearchResult:
            completed = replace(
                result,
                end_to_end_seconds=perf_counter() - started,
                prompt_tokens=usage_delta("prompt_tokens"),
                completion_tokens=usage_delta("completion_tokens"),
                harness_version=self.harness_version,
                memory_hit_ids=tuple(item.memory_id for item in memory_hits),
            )
            emit(
                "result",
                refs=[item.doc_id for item in completed.evidence],
                summary={
                    "status": completed.status,
                    "route": completed.route,
                    "iterations": completed.iterations,
                    "citation_count": len(completed.citations),
                    "refusal_reason": completed.refusal_reason,
                    "memory_hit_ids": completed.memory_hit_ids,
                },
                latency_ms=completed.end_to_end_seconds * 1000,
            )
            return completed

        llm_calls = 0
        if llm_calls >= self.budget.max_llm_calls:
            return finish(
                self._budget_refusal(question, "", [], 0, [], "llm_calls", llm_calls)
            )
        planner_started = perf_counter()
        try:
            research_plan = self.structured_planner.plan(question)
        except (PlannerValidationError, TypeError, ValueError) as exc:
            emit("planner", summary={"valid": False}, error="INVALID_PLAN")
            return finish(ResearchResult(
                question=question,
                route="invalid",
                status="abstained",
                answer="任务规划输出不合法，系统已安全拒答。",
                citations=(),
                evidence=(),
                iterations=0,
                refusal_reason=f"invalid_plan:{exc}",
                llm_calls=1,
            ))
        llm_calls += 1
        plan = research_plan.as_query_plan(question)
        if self.procedural_memory is not None:
            signature = "|".join((plan.route, *plan.company_filters, *plan.metric_filters))
            memory_hits = tuple(self.procedural_memory.search(signature, limit=3))
        emit(
            "planner",
            summary={
                "valid": True,
                "answerability": research_plan.answerability,
                "route": plan.route,
                "entities": plan.company_filters,
                "periods": plan.year_filters,
                "slot_ids": [slot.slot_id for slot in plan.evidence_slots],
                "memory_hit_ids": [item.memory_id for item in memory_hits],
            },
            latency_ms=(perf_counter() - planner_started) * 1000,
        )
        fallback = QueryAnalyzer().analyze(question)
        if not plan.evidence_slots and plan.route not in {"single", "multi"}:
            plan = replace(
                plan,
                sub_questions=plan.sub_questions or plan.sub_queries,
                evidence_slots=fallback.evidence_slots,
                company_filters=plan.company_filters or fallback.company_filters,
                year_filters=plan.year_filters or fallback.year_filters,
                metric_filters=plan.metric_filters or fallback.metric_filters,
            )
        if not self.decompose_queries:
            plan = replace(plan, sub_queries=(question,), sub_questions=(question,))
        if not self.use_evidence_slots:
            plan = replace(plan, evidence_slots=())
        if research_plan.answerability == "unanswerable" or plan.route in {
            "out_of_scope",
            "unanswerable",
        }:
            return finish(ResearchResult(
                question=question,
                route=plan.route,
                status="abstained",
                answer=(
                    "该问题超出金融研报事实核查范围。"
                    if plan.route == "out_of_scope"
                    else "该问题在当前资料与任务边界内不可回答。"
                ),
                citations=(),
                evidence=(),
                iterations=0,
                refusal_reason=(
                    "out_of_scope" if plan.route == "out_of_scope" else "planner_unanswerable"
                ),
                llm_calls=llm_calls,
            ))
        queries = (plan.sub_queries or (question,))[: self.budget.max_sub_queries]
        used_queries: list[str] = []
        accumulated: dict[str, SearchResult] = {}
        provenance: list[tuple[SearchResult, str, int]] = []
        traces: list[AgentTrace] = []

        effective_iterations = self.max_iterations if self.allow_reflection else 1
        for iteration in range(1, effective_iterations + 1):
            round_usage = {
                field: int(dict(getattr(client, "usage", {}) or {}).get(field, 0))
                for field in ("calls", "prompt_tokens", "completion_tokens")
            }
            retrieval_started = perf_counter()
            previous_doc_ids = tuple(accumulated)
            newly_added = self._retrieve(queries, accumulated, provenance, iteration)
            used_queries.extend(queries)
            pack = self.pack_builder.build(
                provenance,
                plan.evidence_slots,
                self.budget,
                expand_neighbors=self.expand_neighbors,
            )
            evidence = pack_to_search_results(pack)
            retrieval_seconds = perf_counter() - retrieval_started
            emit(
                "retrieval",
                refs=[item.doc_id for item in evidence],
                summary={
                    "iteration": iteration,
                    "queries": queries,
                    "new_evidence": newly_added,
                    "slot_coverage": sorted(pack.covered_slots),
                },
                latency_ms=retrieval_seconds * 1000,
            )
            if iteration > 1 and newly_added == 0:
                traces.append(
                    AgentTrace(
                        iteration=iteration,
                        sub_queries=queries,
                        retrieved_doc_ids=tuple(item.doc_id for item in evidence),
                        evidence_sufficient=False,
                        missing_evidence=traces[-1].missing_evidence if traces else (),
                        verification_reasoning="stopped: retrieval added no new evidence",
                        slot_coverage=traces[-1].slot_coverage if traces else (),
                        newly_added_evidence=0,
                        estimated_context_tokens=pack.estimated_tokens,
                        retrieval_seconds=retrieval_seconds,
                    )
                )
                break
            if self.require_verification and llm_calls >= self.budget.max_llm_calls:
                return finish(
                    self._budget_refusal(
                        question, plan.route, evidence, iteration, traces, "llm_calls", llm_calls
                    )
                )
            verifier_started = perf_counter()
            if self.require_verification:
                model_decision = self.llm.verify(question, evidence)
                llm_calls += 1
                if self.use_retrieval_critic:
                    deterministic = self.retrieval_critic.review(plan, pack)
                    decision = merge_decisions(deterministic, model_decision)
                else:
                    decision = model_decision
            else:
                decision = VerificationDecision(
                    True, reasoning="verification disabled by ablation"
                )
            verifier_seconds = perf_counter() - verifier_started
            gate_decision = None
            if not decision.sufficient and iteration < effective_iterations:
                gate_decision = self.retrieval_value_gate.decide(
                    missing_slots=decision.missing_evidence,
                    proposed_queries=decision.follow_up_queries or decision.missing_evidence,
                    used_queries=tuple(used_queries),
                    remaining_rounds=effective_iterations - iteration,
                    previous_doc_ids=previous_doc_ids,
                    current_doc_ids=tuple(accumulated),
                )
                emit(
                    "retrieval_value_gate",
                    summary={
                        "should_retry": gate_decision.should_retry,
                        "missing_slots": gate_decision.expected_missing_slots,
                        "query_delta": gate_decision.query_delta,
                        "stop_reason": gate_decision.stop_reason,
                    },
                )
            traces.append(
                AgentTrace(
                    iteration=iteration,
                    sub_queries=queries,
                    retrieved_doc_ids=tuple(item.doc_id for item in evidence),
                    evidence_sufficient=decision.sufficient,
                    missing_evidence=decision.missing_evidence,
                    verification_reasoning=decision.reasoning,
                    slot_coverage=decision.slot_coverage,
                    follow_up_queries=decision.follow_up_queries,
                    newly_added_evidence=newly_added,
                    estimated_context_tokens=pack.estimated_tokens,
                    retrieval_seconds=retrieval_seconds,
                    verifier_seconds=verifier_seconds,
                    api_calls=int(dict(getattr(client, "usage", {}) or {}).get("calls", 0))
                    - round_usage["calls"],
                    prompt_tokens=int(
                        dict(getattr(client, "usage", {}) or {}).get("prompt_tokens", 0)
                    )
                    - round_usage["prompt_tokens"],
                    completion_tokens=int(
                        dict(getattr(client, "usage", {}) or {}).get("completion_tokens", 0)
                    )
                    - round_usage["completion_tokens"],
                    retry_gate_decision=(
                        "retry" if gate_decision and gate_decision.should_retry else
                        "stop" if gate_decision else "not_applicable"
                    ),
                    retry_stop_reason=gate_decision.stop_reason or "" if gate_decision else "",
                    memory_hit_ids=tuple(item.memory_id for item in memory_hits),
                )
            )

            if decision.sufficient:
                ledger = ()
                issues: tuple[str, ...] = ()
                for repair in range(self.budget.max_answer_repairs + 1):
                    if llm_calls >= self.budget.max_llm_calls:
                        return finish(
                            self._budget_refusal(
                                question, plan.route, evidence, iteration, traces, "llm_calls", llm_calls
                            )
                        )
                    grounded = validate_citations(self.llm.answer(question, evidence), evidence)
                    llm_calls += 1
                    if not grounded.citations:
                        break
                    if self.use_answer_critic:
                        deterministic_valid, ledger, deterministic_issues = self.answer_critic.audit(
                            grounded, evidence
                        )
                        if llm_calls >= self.budget.max_llm_calls:
                            return finish(
                                self._budget_refusal(
                                    question,
                                    plan.route,
                                    evidence,
                                    iteration,
                                    traces,
                                    "llm_calls",
                                    llm_calls,
                                )
                            )
                        validation = self.llm.validate_answer(question, grounded, evidence)
                        llm_calls += 1
                        issues = deterministic_issues + validation.issues
                    else:
                        deterministic_valid = True
                        validation = AnswerValidation(True)
                        issues = ()
                    if deterministic_valid and validation.valid:
                        return finish(ResearchResult(
                            question=question,
                            route=plan.route,
                            status="answered",
                            answer=grounded.answer,
                            citations=grounded.citations,
                            evidence=tuple(evidence),
                            iterations=iteration,
                            traces=tuple(traces),
                            claim_ledger=ledger,
                            llm_calls=llm_calls,
                        ))
                    if repair >= self.budget.max_answer_repairs:
                        return finish(ResearchResult(
                            question=question,
                            route=plan.route,
                            status="abstained",
                            answer="答案语义校验失败，无法生成可靠答案。",
                            citations=(),
                            evidence=tuple(evidence),
                            iterations=iteration,
                            traces=tuple(traces),
                            refusal_reason="answer_grounding_failed: " + "；".join(issues),
                            claim_ledger=ledger,
                            llm_calls=llm_calls,
                        ))
                return finish(ResearchResult(
                    question=question,
                    route=plan.route,
                    status="abstained",
                    answer="证据引用校验失败，无法生成可靠答案。",
                    citations=(),
                    evidence=tuple(evidence),
                    iterations=iteration,
                    traces=tuple(traces),
                    refusal_reason="citation_validation_failed",
                    claim_ledger=ledger,
                    llm_calls=llm_calls,
                ))

            if iteration < effective_iterations:
                if gate_decision is None or not gate_decision.should_retry:
                    break
                queries = gate_decision.follow_up_queries
                queries = queries[: self.budget.max_sub_queries]

        pack = self.pack_builder.build(
            provenance, plan.evidence_slots, self.budget, expand_neighbors=self.expand_neighbors
        )
        evidence = pack_to_search_results(pack)
        missing = traces[-1].missing_evidence if traces else ()
        detail = "、".join(missing) if missing else "回答所需的完整证据"
        return finish(ResearchResult(
            question=question,
            route=plan.route,
            status="abstained",
            answer=f"现有资料不足，无法可靠回答；仍缺少：{detail}。",
            citations=(),
            evidence=tuple(evidence),
            iterations=len(traces),
            traces=tuple(traces),
            refusal_reason="insufficient_evidence",
            llm_calls=llm_calls,
            stop_reason=(traces[-1].retry_stop_reason if traces else ""),
        ))
