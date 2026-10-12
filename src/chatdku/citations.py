"""Give every retrieved passage a number, so an answer can point back at it.

Why this is needed: in a single turn the planner may call a search tool more than
once. If each tool numbered its own results starting from [1], then [1] in the
first observation and [1] in the second would be different passages, and a [1] in
the final answer could not be resolved to anything.

So the whole agent shares one registry, and the registry is cleared at the start
of every turn.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """One retrieved passage."""

    label: int  # The number: 1, 2, 3, ...
    node_id: str  # The passage's id in Qdrant
    source_path: str  # The file it came from


class CitationRegistry:
    """Keeps track of which passages were retrieved this turn, and their numbers."""

    def __init__(self) -> None:
        self._citations: list[Citation] = []  # In the order they were registered
        self._label_by_node_id: dict[str, int] = {}  # Used to avoid duplicates

    def register(self, node_id: str, metadata: dict) -> Citation:
        """Register a passage and return its number.

        Registering the same passage again returns the same number instead of
        issuing a second one. This matters because the vector and keyword tools
        often find the very same passage.
        """
        if node_id in self._label_by_node_id:
            return self._citations[self._label_by_node_id[node_id] - 1]

        source = metadata.get("source_path") or metadata.get("file_name") or "Unknown source"
        citation = Citation(
            label=len(self._citations) + 1,
            node_id=node_id,
            source_path=str(source),
        )
        self._citations.append(citation)
        self._label_by_node_id[node_id] = citation.label
        return citation

    def label_exists(self, label: int) -> bool:
        """Return whether this number was actually issued."""
        return 1 <= label <= len(self._citations)

    def source_list(self, labels: list[int]) -> str:
        """Turn numbers into a Markdown list of sources.

        Numbers are grouped by file, because a 435-page document can supply
        several passages and printing its name once per passage is just noise.
        """
        labels_by_file: dict[str, list[int]] = {}
        for label in sorted(set(labels)):
            if not self.label_exists(label):
                continue
            path = self._citations[label - 1].source_path
            labels_by_file.setdefault(path, []).append(label)

        lines = []
        for path, file_labels in labels_by_file.items():
            numbers = "".join(f"[{n}]" for n in file_labels)
            lines.append(f"- {numbers} {path}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Forget everything, so the next turn starts again from [1]."""
        self._citations.clear()
        self._label_by_node_id.clear()

    def __len__(self) -> int:
        return len(self._citations)
