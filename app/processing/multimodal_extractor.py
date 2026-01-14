from typing import Any, Dict


class MultimodalExtractor:
    """Extracts text, tables, and images from chunks."""

    @staticmethod
    def extract(chunk: Any) -> Dict[str, Any]:
        """Extract text, tables, and images from a chunk.

        Args:
            chunk: A chunk object produced by the chunker. The chunk may
                contain metadata.orig_elements where table and image elements
                are stored.

        Returns:
            A dict with keys:
            - "text": extracted textual content
            - "tables": list of table HTML strings
            - "images": list of base64-encoded image payloads
            - "types": list of detected content types (e.g., "text", "table", "image")
        """
        content = {
            "text": getattr(chunk, "text", ""),
            "tables": [],
            "images": [],
            "types": ["text"],
        }

        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__

                if element_type == "Table":
                    content["types"].append("table")
                    html = getattr(element.metadata, "text_as_html", element.text)
                    content["tables"].append(html)

                elif element_type == "Image":
                    if hasattr(element.metadata, "image_base64"):
                        content["types"].append("image")
                        content["images"].append(element.metadata.image_base64)

        # Deduplicate types
        content["types"] = list(set(content["types"]))
        return content
