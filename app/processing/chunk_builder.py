from typing import List, Any
from unstructured.chunking.title import chunk_by_title


class ChunkBuilder:
    """Creates intelligent chunks using title-based strategy."""

    def __init__(
        self,
        max_chars: int = 3000,
        new_after_chars: int = 2400,
        combine_text_under_n_chars: int = 500
    ):
        self.max_chars = max_chars
        self.new_after_chars = new_after_chars
        self.combine_text_under_n_chars = combine_text_under_n_chars

    def build(self, elements: List[Any]) -> List[Any]:
        """Create chunks from parsed document elements using title heuristics.

        Args:
            elements: A list of parsed elements (from the file loader) to chunk.

        Returns:
            A list of chunk objects suitable for downstream extraction and
            embedding.
        """
        return chunk_by_title(
            elements,
            max_characters=self.max_chars,
            new_after_n_chars=self.new_after_chars,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
        )
