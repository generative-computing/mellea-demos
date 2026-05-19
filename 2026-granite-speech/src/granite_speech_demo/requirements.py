"""Requirement specs for IVR Best-of-N validation.

Each spec encapsulates one validator: a generation-time `instruction` to steer
the model, a `description` for reporting, and a `check()` method that runs the
appropriate ALoRA-backed intrinsic to score the response.

To add a new validator type, subclass `RequirementSpec` and implement `check()`
to return `(passed, score, threshold)`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mellea.stdlib.components.intrinsic import core, guardian

DEFAULT_REQUIREMENT_THRESHOLD = 0.5


@dataclass(frozen=True, kw_only=True)
class RequirementSpec(ABC):
    label: str
    description: str
    instruction: str

    @abstractmethod
    def check(self, gen_ctx, backend) -> tuple[bool, float, float]:
        """Return (passed, score, threshold) for this requirement."""
        ...


@dataclass(frozen=True, kw_only=True)
class RequirementCheckSpec(RequirementSpec):
    """Validates via the requirement-check ALoRA. Returns a float score
    thresholded against `threshold` (or `< threshold` when `invert` is set)."""
    threshold: float = DEFAULT_REQUIREMENT_THRESHOLD
    invert: bool = False

    def check(self, gen_ctx, backend):
        score = core.requirement_check(gen_ctx, backend, self.description)
        passed = score < self.threshold if self.invert else score > self.threshold
        return passed, float(score), self.threshold


@dataclass(frozen=True, kw_only=True)
class PolicyGuardrailSpec(RequirementSpec):
    """Validates via the policy-guardrails ALoRA. Returns a Yes/No/Ambiguous
    label, mapped to a pseudo-score for uniform reporting."""
    policy_text: str
    ambiguous_passes: bool = False

    def check(self, gen_ctx, backend):
        label = guardian.policy_guardrails(gen_ctx, backend, policy_text=self.policy_text)
        passing = {"Yes", "Ambiguous"} if self.ambiguous_passes else {"Yes"}
        passed = label in passing
        score = {"Yes": 1.0, "Ambiguous": 0.5, "No": 0.0}.get(label, 0.0)
        return passed, score, 0.5


IVR_REQUIREMENT_SPECS: list[RequirementSpec] = [
    RequirementCheckSpec(
        label="Natural speech",
        description="The response consists of short, complete sentences that sound natural when spoken aloud.",
        instruction="Use short, complete sentences that sound natural when spoken aloud.",
    ),
    RequirementCheckSpec(
        label="No markdown",
        description="The response contains no bullet points, no numbered lists, no headers, and no markdown formatting.",
        instruction="No bullet points. No numbered lists. No headers. No markdown formatting.",
    ),
    PolicyGuardrailSpec(
        label="Relevant to IBM",
        description="The response is relevant to IBM offerings.",
        instruction="Stay relevant to IBM offerings.",
        policy_text=(
            "Responses must be relevant to IBM products, services, or offerings. "
            "General-knowledge answers unrelated to IBM violate this policy."
        ),
        ambiguous_passes=True,
    ),
    RequirementCheckSpec(
        label="No code",
        description="The response includes software code or pseudocode or offers to help with coding",
        instruction="",
        invert=True,
    ),
]

IVR_REQUIREMENTS = [spec.description for spec in IVR_REQUIREMENT_SPECS]
IVR_REQUIREMENT_LABELS = [spec.label for spec in IVR_REQUIREMENT_SPECS]
IVR_REQUIREMENT_INSTRUCTIONS = [spec.instruction for spec in IVR_REQUIREMENT_SPECS]
