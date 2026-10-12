"""Tests for the citation registry.

No model, no network, no index: the whole file runs in well under a second.
"""

from chatdku.citations import CitationRegistry


def test_first_label_is_one():
    registry = CitationRegistry()
    assert registry.register("node-a", {"source_path": "faq.docx"}).label == 1


def test_labels_are_issued_in_order():
    registry = CitationRegistry()
    first = registry.register("node-a", {"source_path": "faq.docx"})
    second = registry.register("node-b", {"source_path": "bulletin.pdf"})
    assert (first.label, second.label) == (1, 2)


def test_the_same_passage_keeps_one_label():
    registry = CitationRegistry()
    first = registry.register("node-a", {"source_path": "faq.docx"})
    again = registry.register("node-a", {"source_path": "faq.docx"})
    assert first.label == again.label
    assert len(registry) == 1


def test_a_passage_found_by_both_tools_has_one_label():
    # The vector and keyword tools often find the same passage, and it must not
    # end up with two identities in the answer.
    registry = CitationRegistry()
    from_vector = registry.register("shared-chunk", {"source_path": "faq.docx"})
    from_keyword = registry.register("shared-chunk", {"source_path": "faq.docx"})
    assert from_vector.label == from_keyword.label == 1
    assert len(registry) == 1


def test_different_chunks_of_one_file_get_different_labels():
    registry = CitationRegistry()
    first = registry.register("chunk-1", {"source_path": "bulletin.pdf"})
    second = registry.register("chunk-2", {"source_path": "bulletin.pdf"})
    assert first.label != second.label


def test_label_exists_rejects_numbers_that_were_never_issued():
    registry = CitationRegistry()
    registry.register("node-a", {"source_path": "faq.docx"})
    assert registry.label_exists(1) is True
    assert registry.label_exists(2) is False
    assert registry.label_exists(0) is False


def test_source_name_falls_back_to_file_name_then_unknown():
    registry = CitationRegistry()
    assert registry.register("a", {"source_path": "faq.docx"}).source_path == "faq.docx"
    assert registry.register("b", {"file_name": "faq.docx"}).source_path == "faq.docx"
    assert registry.register("c", {}).source_path == "Unknown source"


def test_reset_restarts_numbering():
    registry = CitationRegistry()
    registry.register("node-a", {"source_path": "faq.docx"})
    registry.register("node-b", {"source_path": "faq.docx"})
    registry.reset()
    assert len(registry) == 0
    assert registry.register("node-c", {"source_path": "faq.docx"}).label == 1


def test_source_list_groups_numbers_by_file():
    registry = CitationRegistry()
    first = registry.register("c1", {"source_path": "bulletin.pdf"})
    second = registry.register("c2", {"source_path": "faq.docx"})
    third = registry.register("c3", {"source_path": "bulletin.pdf"})
    assert registry.source_list([first.label, second.label, third.label]) == (
        "- [1][3] bulletin.pdf\n- [2] faq.docx"
    )


def test_source_list_ignores_numbers_that_were_never_issued():
    registry = CitationRegistry()
    registry.register("c1", {"source_path": "faq.docx"})
    assert registry.source_list([1, 42]) == "- [1] faq.docx"


def test_source_list_is_empty_when_nothing_was_cited():
    registry = CitationRegistry()
    registry.register("c1", {"source_path": "faq.docx"})
    assert registry.source_list([]) == ""
    assert registry.source_list([42]) == ""
