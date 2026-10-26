from __future__ import annotations

from .agent_schemas import QueryPlan, VerificationDecision
from .evidence import EvidencePack


class RetrievalCritic:
    """Observable slot-coverage critic independent of any model judgment."""

    def review(self, plan: QueryPlan, pack: EvidencePack) -> VerificationDecision:
        covered = pack.covered_slots
        coverage = tuple(
            (slot.slot_id, "covered" if slot.slot_id in covered else "missing")
            for slot in plan.evidence_slots
        )
        missing_slots = [slot for slot in plan.evidence_slots if slot.slot_id not in covered]
        follow_ups = tuple(slot.description for slot in missing_slots if slot.description)
        unsafe_items = [item.doc_id for item in pack.items if item.contains_untrusted_instruction]
        conflicts = (
            ("untrusted instruction detected in evidence: " + ", ".join(unsafe_items),)
            if unsafe_items
            else ()
        )
        return VerificationDecision(
            sufficient=not missing_slots and not conflicts,
            missing_evidence=tuple(slot.description for slot in missing_slots),
            follow_up_queries=follow_ups,
            reasoning=(
                "all evidence slots covered"
                if not missing_slots and not conflicts
                else (
                    "; ".join(conflicts)
                    if conflicts and not missing_slots
                    else "missing slots: " + ", ".join(slot.slot_id for slot in missing_slots)
                )
            ),
            slot_coverage=coverage,
            conflicts=conflicts,
        )


def merge_decisions(
    deterministic: VerificationDecision,
    model: VerificationDecision,
) -> VerificationDecision:
    """Fail closed: both the slot critic and model verifier must pass."""

    missing = tuple(dict.fromkeys(deterministic.missing_evidence + model.missing_evidence))
    follow_ups = tuple(dict.fromkeys(deterministic.follow_up_queries + model.follow_up_queries))
    return VerificationDecision(
        sufficient=deterministic.sufficient and model.sufficient,
        missing_evidence=missing,
        follow_up_queries=follow_ups,
        reasoning="; ".join(
            part for part in (deterministic.reasoning, model.reasoning) if part
        ),
        slot_coverage=deterministic.slot_coverage or model.slot_coverage,
        conflicts=tuple(dict.fromkeys(deterministic.conflicts + model.conflicts)),
    )
