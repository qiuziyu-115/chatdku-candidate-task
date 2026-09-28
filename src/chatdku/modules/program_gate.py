"""Evidence-scope gate for source-grounded responses."""

from __future__ import annotations

import re

import dspy

from .prompts import PROGRAM_GATE_INSTRUCTIONS


class ProgramGateSignature(dspy.Signature):
    __doc__ = PROGRAM_GATE_INSTRUCTIONS

    current_user_message: str = dspy.InputField()
    trajectory: str = dspy.InputField()
    trajectory_summary: str = dspy.InputField()
    decision: str = dspy.OutputField(desc="Exactly ALLOW, CLARIFY, or INSUFFICIENT.")
    guardrail: str = dspy.OutputField()
    safe_response: str = dspy.OutputField()


class ProgramGate(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(ProgramGateSignature)

    def forward(
        self, current_user_message: str, trajectory: str, trajectory_summary: str = ""
    ) -> dspy.Prediction:
        try:
            prediction = self.predict(
                current_user_message=current_user_message,
                trajectory=trajectory,
                trajectory_summary=trajectory_summary,
            )
            raw = re.sub(r"<think>.*?</think>", "", prediction.decision, flags=re.DOTALL).upper()
            decision = next((item for item in ("ALLOW", "CLARIFY", "INSUFFICIENT") if item in raw), "INSUFFICIENT")
            safe_response = "" if decision == "ALLOW" else prediction.safe_response.strip()
            guardrail = prediction.guardrail.strip()
        except Exception:
            decision = "INSUFFICIENT"
            safe_response = "I cannot confirm the requested details from the currently available sources."
            guardrail = "Do not provide specific requirements, deadlines, contacts, or URLs."
        if decision != "ALLOW" and not safe_response:
            safe_response = "I cannot confirm the requested details from the currently available sources."
        return dspy.Prediction(
            decision=decision,
            guardrail=guardrail,
            safe_response=safe_response,
            allow=decision == "ALLOW",
        )
