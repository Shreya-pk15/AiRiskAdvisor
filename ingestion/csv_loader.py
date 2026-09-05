"""
CSV document loader using pandas.
Converts tabular row data into semantic, human-readable text representations suitable for RAG embedding.
"""

import pandas as pd
from typing import Dict, Any, Union
import io
import os


def load_csv(file_input: Union[str, bytes, io.BytesIO], filename: str = "data.csv") -> Dict[str, Any]:
    """
    Extract text content from a CSV file by converting each row into a structured semantic record.

    Example converted format:
    Task: Payment Integration | Status: In Progress | Deadline: 2026-09-10 | Assignee: John

    Args:
        file_input: File path (str), raw bytes, or BytesIO object.
        filename: Name of the CSV file.

    Returns:
        Dict[str, Any]: {
            "text": Extracted semantic text,
            "source": filename,
            "file_type": "csv"
        }
    """
    try:
        if isinstance(file_input, (bytes, str, io.BytesIO)):
            if isinstance(file_input, bytes):
                df = pd.read_csv(io.BytesIO(file_input))
            elif hasattr(file_input, "read"):
                content = file_input.read()
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                df = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content))
            else:
                df = pd.read_csv(file_input)
        else:
            raise ValueError("Unsupported input type for CSV loader.")

        if df.empty:
            raise ValueError(f"CSV file '{filename}' is empty.")

        row_texts = []
        columns = df.columns.tolist()

        for idx, row in df.iterrows():
            row_items = []
            for col in columns:
                val = row[col]
                # Filter out null or NaN values
                if pd.notna(val) and str(val).strip() != "":
                    row_items.append(f"{col}: {str(val).strip()}")
            
            if row_items:
                row_texts.append(f"Record {idx + 1}: " + " | ".join(row_items))

        full_text = "\n".join(row_texts)

        if not full_text.strip():
            raise ValueError(f"No valid data rows found in CSV file '{filename}'.")

        return {
            "text": full_text,
            "source": os.path.basename(filename),
            "file_type": "csv"
        }
    except Exception as e:
        raise RuntimeError(f"Failed to process CSV file '{filename}': {str(e)}")
