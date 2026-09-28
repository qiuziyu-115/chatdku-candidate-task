"""Multi-step DSPy planner for the public ChatDKU agent harness."""

from __future__ import annotations

import json
from typing import Any

import dspy

from .conversation_memory import ConversationMemory
from .prompts import CHATDKU_ROLE, RETRIEVAL_PLAN_INSTRUCTIONS


class PlannerSignature(dspy.Signature):
    __doc__ = RETRIEVAL_PLAN_INSTRUCTIONS

    current_user_message: str = dspy.InputField()
    conversation_history: str = dspy.InputField()
    conversation_summary: str = dspy.InputField()
    chatbot_role: str = dspy.InputField()


class SummarizerSignature(dspy.Signature):
    __doc__ = """Summarize discarded tool observations while preserving evidence relevant to the user question."""

    current_user_message: str = dspy.InputField()
    trajectory_to_discard: str = dspy.InputField()
    previous_summary: str = dspy.InputField()
    new_summary: str = dspy.OutputField()


class Planner(dspy.Module):
    """Run a bounded sequence of named tools and return its trajectory."""

    def __init__(self, tools: list[Any], max_iterations: int = 5) -> None:
        super().__init__()
        normalized = [tool if isinstance(tool, dspy.Tool) else dspy.Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in normalized}
        self.tools["finish"] = dspy.Tool(
            lambda: "Completed.",
            name="finish",
            desc="Finish after collecting enough information to answer the user.",
            args={},
        )
        tool_list = "\n".join(_describe_tool(name, tool) for name, tool in self.tools.items())
        signature = (
            dspy.Signature({**PlannerSignature.input_fields}, f"{RETRIEVAL_PLAN_INSTRUCTIONS}\n\nAvailable tools:\n{tool_list}")
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=str)
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )
        self.predict = dspy.Predict(signature)
        self.summarizer = dspy.Predict(SummarizerSignature)
        self.max_iterations = max_iterations
        self.trajectory_summary = ""

    def forward(
        self, current_user_message: str, conversation_memory: ConversationMemory
    ) -> dspy.Prediction:
        trajectory: list[dict[str, Any]] = []
        inputs = {
            "current_user_message": current_user_message,
            "conversation_history": conversation_memory.history_str(),
            "conversation_summary": conversation_memory.summary,
            "chatbot_role": CHATDKU_ROLE,
        }
        for _ in range(self.max_iterations):
            plan = self.predict(**inputs, trajectory=_format_trajectory(trajectory))
            name = str(plan.next_tool_name).strip()
            arguments = plan.next_tool_args if isinstance(plan.next_tool_args, dict) else {}
            event: dict[str, Any] = {
                "thought": str(plan.next_thought),
                "tool_name": name,
                "tool_args": arguments,
            }
            if name not in self.tools:
                event["observation"] = f"Unknown tool: {name}. Choose one of {', '.join(self.tools)}."
                trajectory.append(event)
                break
            try:
                event["observation"] = self.tools[name](**arguments)
            except Exception as error:
                event["observation"] = f"Execution error in {name}: {error}"
            trajectory.append(event)
            if len(_format_trajectory(trajectory)) > 16_000 and len(trajectory) > 1:
                discarded = trajectory.pop(0)
                self.trajectory_summary = self.summarizer(
                    current_user_message=current_user_message,
                    trajectory_to_discard=_format_trajectory([discarded]),
                    previous_summary=self.trajectory_summary,
                ).new_summary
            if name == "finish":
                break
        return dspy.Prediction(
            trajectory=_format_trajectory(trajectory), summary=self.trajectory_summary
        )


def _format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    return json.dumps(trajectory, ensure_ascii=False, default=str, indent=2)


def _describe_tool(name: str, tool: dspy.Tool) -> str:
    """Render each callable's exact schema in the planner's instructions."""
    arguments = ", ".join(
        f"{argument_name}: {argument.get('type', 'value')}"
        for argument_name, argument in tool.args.items()
    ) or "no arguments"
    return f"- {name}({arguments}): {tool.desc}"


RetrievalPlanner = Planner
