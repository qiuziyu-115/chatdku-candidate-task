"""Regression tests for the retrieval tool observation format.

The labelled format is what the synthesizer is told to cite, so these tests pin
both the new labelled behaviour and the original unlabelled behaviour.
"""

from chatdku.citations import CitationRegistry
from chatdku.tools.retrieval import RetrievedDocument, format_documents


def _document(node_id: str = "node-1", source_path: str = "faq.docx") -> RetrievedDocument:
    return RetrievedDocument(
        node_id=node_id,
        text="The withdrawal deadline is in week ten.",
        metadata={"source_path": source_path},
        score=0.5,
    )


def test_empty_results_message_is_unchanged():
    assert format_documents([]) == "No matching documents found."


def test_without_a_registry_the_output_keeps_the_original_format():
    rendered = format_documents([_document()])
    assert rendered == (
        "Source: faq.docx\nURL: No URL\nScore: 0.5000\n\n"
        "The withdrawal deadline is in week ten."
    )


def test_with_a_registry_each_passage_is_labelled():
    registry = CitationRegistry()
    rendered = format_documents(
        [_document("node-1"), _document("node-2", source_path="bulletin.pdf")],
        registry,
    )
    assert "[1] Source: faq.docx" in rendered
    assert "[2] Source: bulletin.pdf" in rendered


def test_labels_stay_stable_across_repeated_tool_calls():
    registry = CitationRegistry()
    first = format_documents([_document("node-1")], registry)
    second = format_documents(
        [_document("node-2", source_path="bulletin.pdf"), _document("node-1")], registry
    )
    assert first.startswith("[1] Source: faq.docx")
    # node-1 already owns label 1, so the newly seen passage takes the next one
    # and the repeated passage keeps its original label.
    assert "[2] Source: bulletin.pdf" in second
    assert "[1] Source: faq.docx" in second


def test_registry_grows_by_one_per_new_passage():
    registry = CitationRegistry()
    format_documents([_document("node-1"), _document("node-2")], registry)
    assert len(registry) == 2
