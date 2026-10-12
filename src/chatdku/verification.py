"""Check whether the citations in an answer actually exist.

Why this is needed: the prompt asks the model to cite its sources, but a prompt
is a request, not a guarantee. A small model will sometimes cite a number that
was never issued, or write a whole answer with no citation at all.

The program gate cannot catch either case, because it runs *before* the answer
is written. So the check has to happen here instead.

Everything in this module is a plain function. No model call, no network, so the
tests run in a fraction of a second.
"""

import re

# Matches a citation number such as [1] or [12].
CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")


def parse_citations(text: str) -> list[int]:
    """Return every citation number in the text, in the order it appears."""
    return [int(value) for value in CITATION_PATTERN.findall(text)]


def find_valid_labels(text: str, registry) -> list[int]:
    """Numbers the answer cites that were really issued (sorted, no repeats)."""
    return sorted({label for label in parse_citations(text) if registry.label_exists(label)})


def find_invalid_labels(text: str, registry) -> list[int]:
    """Numbers the answer cites that were never issued, i.e. made up."""
    return sorted({label for label in parse_citations(text) if not registry.label_exists(label)})


def remove_labels(text: str, labels: list[int]) -> str:
    """Delete these numbers from the text.

    Removing a number can leave a stray space or a space before punctuation, so
    tidy those up as well.
    """
    drop = set(labels)

    def keep_or_drop(match: re.Match) -> str:
        return "" if int(match.group(1)) in drop else match.group(0)

    cleaned = CITATION_PATTERN.sub(keep_or_drop, text)
    cleaned = re.sub(r"[ \t]+([.,;:!?。，；：！？])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
