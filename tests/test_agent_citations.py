"""Tests for the agent-side citation checks, with no model call.

``Agent.forward`` cannot run without a model, but the two methods that do the new
work can: they only touch the citation registry and one settings flag. Binding
those real methods onto a small stub keeps the tests fast and still exercises the
submitted code rather than a copy of it.
"""

from types import SimpleNamespace

import pytest

from chatdku.agent import Agent
from chatdku.citations import CitationRegistry

SAFE_RESPONSE = "I cannot confirm that from the available sources."


class _StubAgent:
    """Small object exposing the two real methods under test."""

    _check_citations = Agent._check_citations
    _attach_sources = Agent._attach_sources

    def __init__(self, require_citations: bool = True) -> None:
        self.citations = CitationRegistry()
        self.settings = SimpleNamespace(require_citations=require_citations)


def _gate(allow: bool):
    return SimpleNamespace(allow=allow, safe_response=SAFE_RESPONSE)


@pytest.fixture()
def agent():
    """A turn in which two passages were retrieved."""
    stub = _StubAgent()
    stub.citations.register("node-1", {"source_path": "faq.docx"})
    stub.citations.register("node-2", {"source_path": "bulletin.pdf"})
    return stub


# --- made-up citations -----------------------------------------------------


def test_a_made_up_citation_discards_the_whole_answer(agent):
    # Deleting only the number would leave "The deadline is March 1." in front of
    # the reader: a confident claim whose stated support does not exist.
    response, invalid = agent._check_citations("Real [1]. Made up [42].", _gate(True))
    assert response == SAFE_RESPONSE
    assert invalid == [42]


def test_with_enforcement_off_made_up_numbers_are_still_removed(agent):
    # Measurement mode returns the answer, but never a number that cannot resolve.
    agent.settings = SimpleNamespace(require_citations=False)
    response, invalid = agent._check_citations("Real [1]. Made up [42].", _gate(True))
    assert response == "Real [1]. Made up."
    assert invalid == [42]


def test_a_refused_answer_is_not_replaced_again(agent):
    response, invalid = agent._check_citations(SAFE_RESPONSE, _gate(False))
    assert response == SAFE_RESPONSE
    assert invalid == []


# --- missing citations -----------------------------------------------------


def test_an_allowed_answer_with_no_citation_is_replaced(agent):
    response, invalid = agent._check_citations("A confident answer with no sources.", _gate(True))
    assert response == SAFE_RESPONSE
    assert invalid == []


def test_the_missing_citation_fallback_can_be_switched_off(agent):
    agent.settings = SimpleNamespace(require_citations=False)
    response, invalid = agent._check_citations("A confident answer with no sources.", _gate(True))
    assert response == "A confident answer with no sources."
    assert invalid == []


# --- the source list -------------------------------------------------------


def test_sources_are_built_from_the_registry(agent):
    final = agent._attach_sources("Real [1]. Also real [2].")
    assert final.startswith("Real [1]. Also real [2].")
    assert "**Sources**" in final
    assert "- [1] faq.docx" in final
    assert "- [2] bulletin.pdf" in final


def test_a_file_that_was_retrieved_but_never_cited_is_not_listed(agent):
    final = agent._attach_sources("Only this claim [1].")
    assert "- [1] faq.docx" in final
    assert "bulletin.pdf" not in final


def test_a_made_up_number_is_never_listed_as_a_source(agent):
    # _attach_sources only builds the list; removing numbers from the body is a
    # separate step, so this checks the list itself.
    final = agent._attach_sources("Real [1] and made up [9].")
    source_list = final.split("**Sources**", 1)[1]
    assert "[1] faq.docx" in source_list
    assert "[9]" not in source_list


def test_no_source_list_is_added_when_nothing_is_cited(agent):
    assert agent._attach_sources("No citations here.") == "No citations here."


def test_an_empty_answer_gets_no_source_list(agent):
    assert agent._attach_sources("") == ""


def test_a_discarded_answer_does_not_inherit_a_source_list(agent):
    # Regression test: the numbers must come from the final text, not from an
    # earlier version of the answer that was thrown away.
    response, _ = agent._check_citations("Real [1] and made up [9].", _gate(True))
    assert response == SAFE_RESPONSE
    assert agent._attach_sources(response) == SAFE_RESPONSE


# --- what the user actually sees when an answer is withheld -----------------


def test_an_allowing_gate_with_a_blank_fallback_still_says_something(agent):
    # ProgramGate leaves safe_response empty when it decides ALLOW, because its
    # own message is only needed when it refuses. Reusing that field here
    # returned an empty string to the user. This is the regression test.
    blank_gate = SimpleNamespace(allow=True, safe_response="")
    response, _ = agent._check_citations("A confident answer with no sources.", blank_gate)
    assert response.strip() != ""


def test_a_withheld_answer_from_a_blank_gate_keeps_its_source_list_off(agent):
    blank_gate = SimpleNamespace(allow=True, safe_response="")
    response, _ = agent._check_citations("Real [1] and made up [9].", blank_gate)
    assert response.strip() != ""
    assert agent._attach_sources(response) == response


def test_a_refusing_gate_keeps_its_own_wording(agent):
    refusing = SimpleNamespace(allow=False, safe_response="Ask about a specific program.")
    response, _ = agent._check_citations(refusing.safe_response, refusing)
    assert response == "Ask about a specific program."
