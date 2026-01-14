from typing import List, Any
from unstructured.partition.pdf import partition_pdf

class FileLoader:
    """Loads and parses documents using Unstructured."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_pdf(self) -> List[Any]:
        """Partition the PDF into a sequence of extracted elements.

        Uses the Unstructured library's PDF partitioning to extract text
        blocks, tables (as HTML), and image blocks (exported into the
        element payload as base64). Returns a list of element objects that
        downstream components can chunk and analyze.

        Returns:
            A list of unstructured elements representing the PDF contents.
        """
        return partition_pdf(
            filename=self.file_path,  # Path to your PDF file
            strategy="hi_res",  # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"],  # Grab images found in the PDF
            extract_image_block_to_payload=True  # Store images as base64 data you can actually use
        )
