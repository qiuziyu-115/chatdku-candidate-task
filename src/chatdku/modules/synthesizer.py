"""DSPy module for source-grounded ChatDKU responses."""

from __future__ import annotations

from datetime import date

import dspy

from .prompts import RESPONSE_SYNTHESIS_INSTRUCTIONS


class ResponseSynthesisSignature(dspy.Signature):
    __doc__ = RESPONSE_SYNTHESIS_INSTRUCTIONS

    conversation_history: str = dspy.InputField(desc="Recent conversation history in JSON Lines format.")
    conversation_summary: str = dspy.InputField(desc="Summary of older conversation history.")
    gate_instruction: str = dspy.InputField(desc="Authoritative instruction from the program gate.")
    trajectory: str = dspy.InputField(desc="Tool calls and retrieved observations.")
    trajectory_summary: str = dspy.InputField(desc="Summary of truncated tool history.")
    current_date: str = dspy.InputField(desc="Today's date in ISO format.")
    current_user_message: str = dspy.InputField(desc="The user's original question.")
    response: str = dspy.OutputField(
        desc="A clear answer grounded in the supplied context."
    )


class ResponseSynthesizer(dspy.Module):
    """Produce a grounded final answer from retrieved document passages."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(ResponseSynthesisSignature)

    def forward(
        self,
        current_user_message: str,
        *,
        conversation_history: str = "",
        conversation_summary: str = "",
        gate_instruction: str = "Use only the evidence directly relevant to the question.",
        trajectory: str = "",
        trajectory_summary: str = "",
        current_date: date | None = None,
    ) -> dspy.Prediction:
        return self.predict(
            current_user_message=current_user_message.strip(),
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
            gate_instruction=gate_instruction,
            trajectory=trajectory,
            trajectory_summary=trajectory_summary,
            current_date=(current_date or date.today()).isoformat(),
        )


Synthesizer = ResponseSynthesizer
SynthesizerSignature = ResponseSynthesisSignature
