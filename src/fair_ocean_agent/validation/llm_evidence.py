"""Independent LLM support checking for explicit text-extracted RawFacts.

The extractor only proves that a model cited a real source segment ID and
that Python copied the cited segment into evidence_quote. This verifier is
the semantic second opinion: does that quote actually support the fact
type and raw value?
"""
from __future__ import annotations

from dataclasses import dataclass

from fair_ocean_agent.database.enums import ValidationSeverity, ValidationStatus
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.validation.logical import ValidationOutcome

VERIFIER_VERSION = "llm-evidence-support-v1"

PROMPT_TEMPLATE = """You are checking evidence for one extracted scientific metadata fact.

Decide whether the evidence quote explicitly supports the fact_type_candidate and raw_value.

Rules:
- Answer supported=true only when the quote directly states the value, or states wording that is plainly equivalent.
- Answer supported=false if the quote does not mention the value, contradicts it, only implies it weakly, or supports a different fact type.
- Do not use outside knowledge.
- Do not reward likely or typical lab practice.
- Return ONLY JSON with:
  - supported (boolean)
  - reason (short string)

fact_type_candidate: {fact_type_candidate}
raw_value: {raw_value}
evidence_quote:
\"\"\"
{evidence_quote}
\"\"\"
"""


@dataclass(frozen=True)
class LlmEvidenceSupportResult:
    outcome: ValidationOutcome
    model_name: str


def build_support_check_prompt(fact_type_candidate: str, raw_value: str, evidence_quote: str) -> str:
    return PROMPT_TEMPLATE.format(
        fact_type_candidate=fact_type_candidate,
        raw_value=raw_value,
        evidence_quote=evidence_quote,
    )


def verify_fact_support_with_llm(
    backend: LLMBackend,
    *,
    fact_type_candidate: str,
    raw_value: str,
    evidence_quote: str,
    max_tokens: int | None = None,
) -> LlmEvidenceSupportResult:
    prompt = build_support_check_prompt(fact_type_candidate, raw_value, evidence_quote)
    try:
        parsed, response = backend.generate_json(prompt, temperature=0, max_tokens=max_tokens)
    except LLMBackendError as exc:
        return LlmEvidenceSupportResult(
            ValidationOutcome(
                ValidationStatus.NOT_ASSESSED.value,
                ValidationSeverity.WARNING.value,
                f"LLM evidence verifier failed: {exc}",
                {"error": str(exc)},
            ),
            backend.label,
        )

    if not isinstance(parsed, dict) or not isinstance(parsed.get("supported"), bool):
        return LlmEvidenceSupportResult(
            ValidationOutcome(
                ValidationStatus.NOT_ASSESSED.value,
                ValidationSeverity.WARNING.value,
                "LLM evidence verifier did not return a supported boolean.",
                {"raw_response": response.text if response else None},
            ),
            backend.label,
        )

    supported = parsed["supported"]
    reason = str(parsed.get("reason") or "").strip()
    status = ValidationStatus.SUPPORTED.value if supported else ValidationStatus.UNSUPPORTED.value
    severity = ValidationSeverity.INFO.value if supported else ValidationSeverity.ERROR.value
    message = reason or ("Evidence supports the fact." if supported else "Evidence does not support the fact.")
    return LlmEvidenceSupportResult(
        ValidationOutcome(
            status,
            severity,
            message,
            {
                "supported": supported,
                "reason": reason,
                "verifier_model": response.model if response else backend.label,
            },
        ),
        response.model if response else backend.label,
    )
