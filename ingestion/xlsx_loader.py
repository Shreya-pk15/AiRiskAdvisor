"""
Excel (.xlsx) document loader using openpyxl via pandas.
Converts all sheets in the workbook into semantic, human-readable text representations
suitable for RAG chunking and embedding.
"""

import pandas as pd
from typing import Dict, Any, Union
import io
import os


def load_xlsx(file_input: Union[str, bytes, io.BytesIO], filename: str = "data.xlsx") -> Dict[str, Any]:
    """
    Extract text content from an Excel (.xlsx) file.

    - Reads ALL sheets in the workbook.
    - Each sheet is converted to structured semantic records (same format as CSV loader):
      Record 1: Column A: Value | Column B: Value | ...
    - Sheets are separated by a header line indicating the sheet name.

    Args:
        file_input: File path (str), raw bytes, or BytesIO object.
        filename: Name of the Excel file.

    Returns:
        Dict[str, Any]: {
            "text": Extracted semantic text across all sheets,
            "source": filename,
            "file_type": "xlsx"
        }
    """
    try:
        # Normalise input to a BytesIO stream
        if isinstance(file_input, bytes):
            buffer = io.BytesIO(file_input)
        elif hasattr(file_input, "read"):
            content = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            buffer = io.BytesIO(content if isinstance(content, bytes) else content.encode("utf-8"))
        elif isinstance(file_input, str):
            # Treat as file path
            buffer = file_input
        else:
            raise ValueError("Unsupported input type for Excel loader.")

        # Read all sheets
        excel_data = pd.read_excel(buffer, sheet_name=None, engine="openpyxl")

        if not excel_data:
            raise ValueError(f"Excel file '{filename}' contains no sheets.")

        all_sheet_texts = []

        for sheet_name, df in excel_data.items():
            if df.empty:
                continue

            # Drop completely empty rows
            df = df.dropna(how="all")
            if df.empty:
                continue

            # Normalise column headers (fill unnamed headers)
            df.columns = [
                str(col).strip() if not str(col).startswith("Unnamed") else f"Column_{i+1}"
                for i, col in enumerate(df.columns)
            ]

            row_texts = []
            for idx, row in df.iterrows():
                row_items = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() not in ("", "nan", "NaT", "None"):
                        row_items.append(f"{col}: {str(val).strip()}")
                if row_items:
                    row_texts.append(f"Row {idx + 1}: " + " | ".join(row_items))

            if row_texts:
                sheet_header = f"=== Sheet: {sheet_name} ==="
                all_sheet_texts.append(sheet_header + "\n" + "\n".join(row_texts))

        full_text = "\n\n".join(all_sheet_texts)

        if not full_text.strip():
            raise ValueError(f"No valid data found in Excel file '{filename}'.")

        return {
            "text": full_text,
            "source": os.path.basename(filename),
            "file_type": "xlsx",
        }

    except Exception as e:
        raise RuntimeError(f"Failed to process Excel file '{filename}': {str(e)}")
