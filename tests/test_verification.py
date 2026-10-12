"""Tests for citation verification.

These cover the two things the program gate structurally cannot check, because
the gate runs before the answer exists.
"""

import pytest

from chatdku.citations import CitationRegistry
from chatdku.verification import (
    find_invalid_labels,
    find_valid_labels,
    parse_citations,
    remove_labels,
)


@pytest.fixture()
def registry():
    """A turn in which two passages were retrieved: [1] and [2]."""
    reg = CitationRegistry()
    reg.register("node-1", {"source_path": "faq.docx"})
    reg.register("node-2", {"source_path": "bulletin.pdf"})
    return reg


# --- reading the numbers ---------------------------------------------------


def test_parse_citations_returns_every_number_in_order():
    assert parse_citations("A [1] and B [2] and C [1].") == [1, 2, 1]


@pytest.mark.parametrize("text", ["[abc]", "[]", "[1234]", "[1.5]", "no numbers here", "[ 1 ]"])
def test_parse_citations_ignores_malformed_brackets(text):
    assert parse_citations(text) == []


# --- telling real numbers from made-up ones --------------------------------


def test_real_and_made_up_numbers_are_separated(registry):
    text = "A real one [1], another real one [2], and a made-up [7]."
    assert find_valid_labels(text, registry) == [1, 2]
    assert find_invalid_labels(text, registry) == [7]


def test_citing_the_same_number_twice_counts_once(registry):
    assert find_valid_labels("First point [1]. Second point [1].", registry) == [1]


def test_an_answer_with_no_citations_reports_nothing(registry):
    text = "This answer cites nothing at all."
    assert find_valid_labels(text, registry) == []
    assert find_invalid_labels(text, registry) == []


def test_label_zero_counts_as_made_up(registry):
    # Numbering starts at 1, so [0] was never issued.
    assert find_invalid_labels("It cites [0].", registry) == [0]


# --- removing numbers ------------------------------------------------------


def test_remove_labels_keeps_the_numbers_that_were_not_listed(registry):
    assert remove_labels("Keep [1], drop [7].", [7]) == "Keep [1], drop."


def test_remove_labels_tidies_the_space_before_punctuation(registry):
    assert remove_labels("A sentence [9].", [9]) == "A sentence."


def test_remove_labels_changes_nothing_when_the_list_is_empty(registry):
    text = "Keep [1] and [2]."
    assert remove_labels(text, []) == text
