from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

from .agent_schemas import Claim, GroundedAnswer
from .schemas import SearchResult

_SPLIT_CLAIM = re.compile(r"(?<=[。！？；;])")
_YEAR = re.compile(r"(?:19|20)\d{2}")
_VALUE_UNIT = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元|%|人|辆|GWh|吨)?")
_COMPANIES = ("比亚迪", "宁德时代", "BYD", "CATL")
_METRICS = ("营业收入", "研发投入", "研发费用", "净利润", "毛利率", "销量", "产能")


def decimal_round(value: str | Decimal, places: int = 2) -> Decimal:
    quantum = Decimal(1).scaleb(-places)
    return Decimal(str(value).replace(",", "")).quantize(quantum, rounding=ROUND_HALF_UP)


class AnswerCritic:
    """Build an auditable claim ledger and reject unsupported atomic claims."""

    def audit(
        self,
        answer: GroundedAnswer,
        evidence: list[SearchResult],
    ) -> tuple[bool, tuple[Claim, ...], tuple[str, ...]]:
        sources = {item.doc_id: item for item in evidence}
        citations = {
            citation.doc_id: citation
            for citation in answer.citations
            if citation.doc_id in sources and citation.quote
        }
        if answer.claims:
            raw_claims = [
                (claim.claim_id, claim.text.strip(), claim.citation_ids)
                for claim in answer.claims
                if claim.text.strip()
            ]
        else:
            raw_claims = [
                (f"c{index}", part.strip(), ())
                for index, part in enumerate(_SPLIT_CLAIM.split(answer.answer), 1)
                if part.strip()
            ]
        ledger: list[Claim] = []
        issues: list[str] = []
        for claim_id, text, declared_ids in raw_claims:
            explicit_ids = list(declared_ids) or [
                citation.doc_id
                for citation in answer.citations
                if not citation.claim_ids or claim_id in citation.claim_ids
            ]
            supporting_ids: list[str] = []
            claim_values = {
                (match.group(1).replace(",", ""), match.group(2) or "")
                for match in _VALUE_UNIT.finditer(text)
            }
            claim_years = set(_YEAR.findall(text))
            claim_companies = {name for name in _COMPANIES if name.lower() in text.lower()}
            claim_metrics = {metric for metric in _METRICS if metric in text}
            for doc_id in explicit_ids:
                citation = citations.get(doc_id)
                source = sources.get(doc_id)
                if citation is None or source is None:
                    continue
                normalized_quote = re.sub(r"\s+", "", citation.quote)
                if normalized_quote not in re.sub(r"\s+", "", source.text):
                    continue
                quote_values = {
                    (match.group(1).replace(",", ""), match.group(2) or "")
                    for match in _VALUE_UNIT.finditer(citation.quote)
                }
                quote_years = set(_YEAR.findall(citation.quote))
                if claim_values and not claim_values.issubset(quote_values):
                    continue
                if claim_years and quote_years and not claim_years.issubset(quote_years):
                    continue
                if (
                    claim_years
                    and not quote_years
                    and source.report_year is not None
                    and claim_years != {str(source.report_year)}
                ):
                    continue
                source_text = f"{source.title}\n{source.text}".lower()
                if claim_companies and not all(
                    name.lower() in source_text or name == source.company
                    for name in claim_companies
                ):
                    continue
                if claim_metrics and not all(metric in source_text for metric in claim_metrics):
                    continue
                supporting_ids.append(doc_id)
            supported = bool(supporting_ids) or not text
            claim_issues = () if supported else ("no citation supports this atomic claim",)
            if not supported:
                issues.append(f"{claim_id}: {claim_issues[0]}")
            first_value = next(_VALUE_UNIT.finditer(text), None)
            ledger.append(
                Claim(
                    claim_id=claim_id,
                    text=text,
                    year=int(next(iter(claim_years))) if len(claim_years) == 1 else None,
                    value=first_value.group(1) if first_value else "",
                    unit=(first_value.group(2) or "") if first_value else "",
                    citation_ids=tuple(supporting_ids),
                    supported=supported,
                    issues=claim_issues,
                )
            )
        return not issues, tuple(ledger), tuple(issues)
