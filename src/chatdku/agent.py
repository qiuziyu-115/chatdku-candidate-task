"""The public ChatDKU agent harness."""

from __future__ import annotations

import json
import logging
from typing import Any

# DSPy installs lazy imports for optional packages. Qdrant imports
# ``numpy.typing`` during startup, so load NumPy eagerly before DSPy to avoid
# its lazy-import proxy interfering with NumPy's internal module setup.
import numpy  # noqa: F401
import dspy

from .config import Settings, get_settings
from .modules import (
    ConversationMemory,
    Planner,
    ProgramGate,
    QueryRewrite,
    Synthesizer,
    ToolMemory,
)
from .tools import KeywordRetriever, VectorRetriever


logger = logging.getLogger(__name__)


def configure_dspy(harness_settings: Settings | None = None) -> Settings:
    """Configure DSPy for the selected OpenAI-compatible chat endpoint."""
    active_settings = harness_settings or get_settings()
    if not active_settings.llm_model:
        raise ValueError("Set CHATDKU_LLM_MODEL before running the agent.")
    model = active_settings.llm_model
    if not model.startswith("openai/"):
        model = f"openai/{model}"
    dspy.configure(
        lm=dspy.LM(
            model=model,
            api_base=active_settings.llm_base_url,
            api_key=active_settings.llm_api_key or "not-required",
            model_type="chat",
            max_tokens=active_settings.llm_max_tokens,
            temperature=active_settings.llm_temperature,
        )
    )
    return active_settings


class Agent(dspy.Module):
    """Run ChatDKU's retrieval, evidence-gating, and response-synthesis loop."""

    def __init__(
        self,
        *,
        max_iterations: int = 3,
        rewrite_query: bool = True,
        tools: list[Any] | None = None,
        harness_settings: Settings | None = None,
    ) -> None:
        super().__init__()
        self.settings = configure_dspy(harness_settings)
        self.rewrite_query = rewrite_query
        self.conversation_memory = ConversationMemory()
        self.tool_memory = ToolMemory()
        self.query_rewriter = QueryRewrite()
        self.program_gate = ProgramGate()
        self.synthesizer = Synthesizer()
        self.vector_retriever = VectorRetriever(self.settings)
        self.keyword_retriever = KeywordRetriever(self.settings)
        self.planner = Planner(tools or [self.vector_search, self.keyword_search], max_iterations)
        self.previous_response: str | None = None

    def reset(self) -> None:
        self.conversation_memory = ConversationMemory()
        self.tool_memory.reset()
        self.previous_response = None

    def vector_search(self, semantic_query: str, top_k: int | None = None) -> str:
        """Search indexed DKU documents by semantic similarity.

        Args:
            semantic_query: A natural-language query for conceptual searches.
            top_k: Optional maximum number of results to return.
        """
        return self.vector_retriever(semantic_query, top_k)

    def keyword_search(self, keyword_query: str, top_k: int | None = None) -> str:
        """Search indexed DKU documents for exact words, names, phrases, and codes.

        Args:
            keyword_query: Specific terms or phrases for BM25 keyword matching.
            top_k: Optional maximum number of results to return.
        """
        return self.keyword_retriever(keyword_query, top_k)

    def forward(self, current_user_message: str) -> dspy.Prediction:
        """Answer one question and retain the completed turn as conversation memory."""
        if self.previous_response:
            self.conversation_memory("assistant", self.previous_response)

        logger.debug("Agent received user message:\n%s", current_user_message)
        planner_message = current_user_message
        if self.rewrite_query:
            rewrite_context = {
                "current_user_message": current_user_message,
                "conversation_history": self.conversation_memory.history_str(),
                "conversation_summary": self.conversation_memory.summary,
                "tool_history": self.tool_memory.history_str(),
                "tool_summary": self.tool_memory.summary,
            }
            _log_context("query rewrite", rewrite_context)
            rewrite = self.query_rewriter(
                current_user_message,
                conversation_history=rewrite_context["conversation_history"],
                conversation_summary=rewrite_context["conversation_summary"],
                tool_history=rewrite_context["tool_history"],
                tool_summary=rewrite_context["tool_summary"],
            )
            planner_message = rewrite.rewritten_query
            logger.debug("Query rewrite result:\n%s", planner_message)

        _log_context(
            "planner",
            {
                "current_user_message": planner_message,
                "conversation_history": self.conversation_memory.history_str(),
                "conversation_summary": self.conversation_memory.summary,
            },
        )
        plan = self.planner(planner_message, self.conversation_memory)
        logger.debug("Planner trajectory (tool calls and observations):\n%s", plan.trajectory)
        logger.debug("Planner trajectory summary:\n%s", plan.summary)
        self._record_tool_history(current_user_message, plan.trajectory)
        gate = self.program_gate(current_user_message, plan.trajectory, plan.summary)
        logger.debug(
            "Program gate decision: %s\nGuardrail: %s\nSafe response: %s",
            gate.decision,
            gate.guardrail,
            gate.safe_response,
        )

        if gate.allow:
            synthesis_context = {
                "current_user_message": current_user_message,
                "conversation_history": self.conversation_memory.history_str(),
                "conversation_summary": self.conversation_memory.summary,
                "gate_instruction": gate.guardrail,
                "trajectory": plan.trajectory,
                "trajectory_summary": plan.summary,
            }
            _log_context("response synthesis", synthesis_context)
            response = self.synthesizer(
                current_user_message,
                conversation_history=synthesis_context["conversation_history"],
                conversation_summary=synthesis_context["conversation_summary"],
                gate_instruction=synthesis_context["gate_instruction"],
                trajectory=synthesis_context["trajectory"],
                trajectory_summary=synthesis_context["trajectory_summary"],
            ).response
        else:
            response = gate.safe_response

        logger.debug("Final agent response:\n%s", response)
        self.conversation_memory("user", current_user_message)
        self.previous_response = response
        return dspy.Prediction(
            response=response,
            rewritten_query=planner_message,
            trajectory=plan.trajectory,
            gate_decision=gate.decision,
        )

    def _record_tool_history(self, current_user_message: str, trajectory: str) -> None:
        """Record completed tool calls without exposing them in the final answer."""
        try:
            events = json.loads(trajectory)
        except json.JSONDecodeError:
            return
        for event in events:
            name = event.get("tool_name")
            if name and name != "finish":
                self.tool_memory(
                    current_user_message,
                    self.conversation_memory,
                    name,
                    event.get("tool_args", {}),
                    event.get("observation", ""),
                )


def _log_context(stage: str, context: dict[str, str]) -> None:
    """Emit the exact non-secret inputs supplied to one agent stage at DEBUG."""
    logger.debug(
        "%s context window:\n%s",
        stage.capitalize(),
        json.dumps(context, ensure_ascii=False, indent=2, default=str),
    )
