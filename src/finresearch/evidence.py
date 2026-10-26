from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .agent_schemas import EvidenceSlot
from .budgets import ResearchBudget, estimate_tokens
from .schemas import Document, SearchResult

_UNTRUSTED_INSTRUCTION_PATTERNS = (
    re.compile(r"ignore (?:all |any )?(?:previous|prior|system) instructions?", re.IGNORECASE),
    re.compile(r"disregard (?:the )?(?:previous|system) (?:prompt|instructions?)", re.IGNORECASE),
    re.compile(r"忽略(?:之前|以上|系统)(?:的)?(?:指令|提示|要求)"),
    re.compile(r"你现在是.{0,24}(?:助手|系统|模型)"),
)


@dataclass(frozen=True)
class EvidenceItem:
    doc_id: str
    company: str
    year: int | None
    page: int | None
    section: str
    chunk_id: str
    text: str
    score: float
    source_query: str
    covered_slots: tuple[str, ...] = ()
    adjacent_to: tuple[str, ...] = ()
    retrieval_round: int = 1
    duplicate_group: str = ""
    contains_untrusted_instruction: bool = False


def contains_untrusted_instruction(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _UNTRUSTED_INSTRUCTION_PATTERNS)


@dataclass(frozen=True)
class EvidencePack:
    items: tuple[EvidenceItem, ...] = field(default_factory=tuple)

    @property
    def covered_slots(self) -> frozenset[str]:
        return frozenset(slot for item in self.items for slot in item.covered_slots)

    @property
    def estimated_tokens(self) -> int:
        return sum(estimate_tokens(item.text) for item in self.items)


def _normalized_duplicate_key(text: str) -> str:
    normalized = re.sub(r"[\s,，。；;:]", "", text).lower()
    return normalized[:240]


def _slot_matches(slot: EvidenceSlot, result: SearchResult) -> bool:
    haystack = f"{result.title}\n{result.text}".lower()
    if slot.company and slot.company.lower() not in haystack and slot.company != result.company:
        return False
    if slot.year is not None and str(slot.year) not in haystack and slot.year != result.report_year:
        return False
    return not slot.metric or slot.metric.lower() in haystack


class EvidencePackBuilder:
    def __init__(self, documents: list[Document] | None = None) -> None:
        corpus = documents or []
        linked: list[Document] = []
        for index, item in enumerate(corpus):
            previous = corpus[index - 1] if index > 0 else None
            following = corpus[index + 1] if index + 1 < len(corpus) else None
            linked.append(
                replace(
                    item,
                    prev_chunk_id=item.prev_chunk_id
                    or (
                        previous.doc_id
                        if previous is not None and previous.document_id == item.document_id
                        else ""
                    ),
                    next_chunk_id=item.next_chunk_id
                    or (
                        following.doc_id
                        if following is not None and following.document_id == item.document_id
                        else ""
                    ),
                )
            )
        self.documents = {item.doc_id: item for item in linked}

    def build(
        self,
        retrieved: list[tuple[SearchResult, str, int]],
        slots: tuple[EvidenceSlot, ...],
        budget: ResearchBudget,
        *,
        expand_neighbors: bool = True,
    ) -> EvidencePack:
        by_id: dict[str, EvidenceItem] = {}
        for result, query, retrieval_round in retrieved:
            item = self._from_result(result, query, retrieval_round, slots)
            existing = by_id.get(item.doc_id)
            if existing is None or item.score > existing.score:
                by_id[item.doc_id] = item

        if expand_neighbors and budget.max_neighbor_chunks:
            self._expand_neighbors(by_id, slots, budget)
            self._propagate_adjacent_coverage(by_id, slots)

        ordered = sorted(by_id.values(), key=lambda item: (-item.score, item.doc_id))
        selected: list[EvidenceItem] = []
        tokens = 0
        for item in ordered:
            item_tokens = estimate_tokens(item.text)
            if len(selected) >= budget.max_evidence_chunks:
                break
            if tokens + item_tokens > budget.max_context_tokens:
                continue
            selected.append(item)
            tokens += item_tokens
        return EvidencePack(tuple(selected))

    def _from_result(
        self,
        result: SearchResult,
        query: str,
        retrieval_round: int,
        slots: tuple[EvidenceSlot, ...],
        *,
        adjacent_to: tuple[str, ...] = (),
    ) -> EvidenceItem:
        return EvidenceItem(
            doc_id=result.doc_id,
            company=result.company,
            year=result.report_year,
            page=result.page_start,
            section=result.section_path,
            chunk_id=result.doc_id,
            text=result.text,
            score=result.score,
            source_query=query,
            covered_slots=tuple(slot.slot_id for slot in slots if _slot_matches(slot, result)),
            adjacent_to=adjacent_to,
            retrieval_round=retrieval_round,
            duplicate_group=_normalized_duplicate_key(result.text),
            contains_untrusted_instruction=contains_untrusted_instruction(result.text),
        )

    def _expand_neighbors(
        self,
        by_id: dict[str, EvidenceItem],
        slots: tuple[EvidenceSlot, ...],
        budget: ResearchBudget,
    ) -> None:
        seeds = list(by_id.values())
        added = 0
        for seed in seeds:
            source = self.documents.get(seed.doc_id)
            if source is None:
                continue
            for neighbor_id in (source.prev_chunk_id, source.next_chunk_id):
                if not neighbor_id or neighbor_id in by_id or added >= budget.max_neighbor_chunks:
                    continue
                neighbor = self.documents.get(neighbor_id)
                if neighbor is None:
                    continue
                result = SearchResult(
                    doc_id=neighbor.doc_id,
                    score=seed.score,
                    rank=0,
                    title=neighbor.title,
                    text=neighbor.text,
                    source=neighbor.source,
                    document_id=neighbor.document_id,
                    company=neighbor.company,
                    report_year=neighbor.report_year,
                    page_start=neighbor.page_start,
                    page_end=neighbor.page_end,
                    section_path=neighbor.section_path,
                    prev_chunk_id=neighbor.prev_chunk_id,
                    next_chunk_id=neighbor.next_chunk_id,
                )
                by_id[neighbor_id] = self._from_result(
                    result,
                    seed.source_query,
                    seed.retrieval_round,
                    slots,
                    adjacent_to=(seed.doc_id,),
                )
                added += 1

    @staticmethod
    def _propagate_adjacent_coverage(
        by_id: dict[str, EvidenceItem], slots: tuple[EvidenceSlot, ...]
    ) -> None:
        """Let a table header/title and its adjacent data row share slot context."""

        for item in list(by_id.values()):
            for seed_id in item.adjacent_to:
                seed = by_id.get(seed_id)
                if seed is None:
                    continue
                combined_result = SearchResult(
                    doc_id=seed.doc_id,
                    score=max(seed.score, item.score),
                    rank=0,
                    text=f"{seed.text}\n{item.text}",
                    company=seed.company or item.company,
                    report_year=seed.year or item.year,
                    section_path=seed.section or item.section,
                )
                inferred = tuple(
                    slot.slot_id for slot in slots if _slot_matches(slot, combined_result)
                )
                combined = tuple(
                    dict.fromkeys(seed.covered_slots + item.covered_slots + inferred)
                )
                by_id[seed_id] = replace(seed, covered_slots=combined)
                by_id[item.doc_id] = replace(item, covered_slots=combined)


def pack_to_search_results(pack: EvidencePack) -> list[SearchResult]:
    return [
        SearchResult(
            doc_id=item.doc_id,
            score=item.score,
            rank=index,
            text=item.text,
            company=item.company,
            report_year=item.year,
            page_start=item.page,
            page_end=item.page,
            section_path=item.section,
        )
        for index, item in enumerate(pack.items, 1)
    ]
