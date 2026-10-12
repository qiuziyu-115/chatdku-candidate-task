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

from .citations import CitationRegistry
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
from .verification import find_invalid_labels, find_valid_labels, remove_labels


logger = logging.getLogger(__name__)

# Used when an answer has to be withheld but the gate itself did not refuse.
# The gate only fills in ``safe_response`` when it says ALLOW is impossible;
# for an ALLOW it is empty by design, so it cannot be reused here.
WITHHELD_RESPONSE = (
    "I found related documents but could not produce an answer whose sources "
    "can be verified. Please rephrase the question, or ask about a more "
    "specific topic."
)


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
        # One registry per turn, shared by every retrieval tool, so a label
        # means the same passage no matter which tool produced the observation.
        self.citations = CitationRegistry()
        self.vector_retriever = VectorRetriever(self.settings, registry=self.citations)
        self.keyword_retriever = KeywordRetriever(self.settings, registry=self.citations)
        self.planner = Planner(tools or [self.vector_search, self.keyword_search], max_iterations)
        self.previous_response: str | None = None

    def reset(self) -> None:
        self.conversation_memory = ConversationMemory()
        self.tool_memory.reset()
        self.citations.reset()
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

        # Labels are only meaningful inside the turn that issued them, so this
        # turn starts numbering from [1] again.
        self.citations.reset()

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

        response, invalid_labels = self._check_citations(response, gate)
        response = self._attach_sources(response)

        logger.debug("Final agent response:\n%s", response)
        self.conversation_memory("user", current_user_message)
        self.previous_response = response
        return dspy.Prediction(
            response=response,
            rewritten_query=planner_message,
            trajectory=plan.trajectory,
            gate_decision=gate.decision,
            invalid_labels=invalid_labels,
        )

    def _check_citations(self, response: str, gate: dspy.Prediction) -> tuple[str, list[int]]:
        """Check the answer and fall back to the safe response when it fails.

        Two things are checked here, and the program gate can check neither of
        them because it runs before the answer exists:

        1. Was every number in the answer really issued? Making one up means
           claiming support that does not exist. Deleting just the number would
           leave the sentence behind looking clean but unsupported -- for example
           "The deadline is March 1." with its false citation quietly removed --
           so the whole answer is thrown away instead.
        2. The gate said the evidence was good enough, yet the answer cites
           nothing at all. That contradicts the gate, so the answer goes too.

        Returns the final answer and the list of made-up numbers, which the
        evaluation scripts read for their statistics.
        """
        invalid = find_invalid_labels(response, self.citations)
        enforcing = self.settings.require_citations and gate.allow

        if not enforcing:
            # Either enforcement is switched off, or the gate already refused.
            # The answer is returned for measurement, but a reader should still
            # never see a number that cannot be looked up.
            if invalid:
                logger.warning("Answer cited numbers that were never issued: %s. Removing them.", invalid)
                response = remove_labels(response, invalid)
            return response, invalid

        if invalid:
            logger.warning("Answer cited numbers that were never issued: %s. Discarding the answer.", invalid)
            return _withheld_response(gate), invalid

        if not find_valid_labels(response, self.citations):
            logger.warning("Gate allowed the answer, but it cited nothing. Withholding it.")
            return _withheld_response(gate), invalid

        return response, invalid


    def _attach_sources(self, response: str) -> str:
        """Append the list of sources to the answer.

        Which file a number refers to is information the program already holds,
        so there is no reason to ask the model to write it out again. The numbers
        are read back out of the final text, which means the list can never
        describe an answer that was already thrown away.
        """
        if not response:
            return response
        labels = find_valid_labels(response, self.citations)
        if not labels:
            return response
        return f"{response}\n\n---\n\n**Sources**\n{self.citations.source_list(labels)}"

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


def _withheld_response(gate: dspy.Prediction) -> str:
    """Pick the message to return when an answer cannot be published.

    A refusing gate supplies its own wording. An allowing gate leaves
    ``safe_response`` empty, so falling back to it would show the user
    nothing at all -- which is how this was first written, and how running
    the evaluation caught it.
    """
    return gate.safe_response or WITHHELD_RESPONSE


def _log_context(stage: str, context: dict[str, str]) -> None:
    """Emit the exact non-secret inputs supplied to one agent stage at DEBUG."""
    logger.debug(
        "%s context window:\n%s",
        stage.capitalize(),
        json.dumps(context, ensure_ascii=False, indent=2, default=str),
    )
