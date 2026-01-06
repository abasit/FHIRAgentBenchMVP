import os
import pandas as pd
from .registry import tool_registry
from .cache import cached_tool

# Get the directory where this file is located
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.normpath(os.path.join(TOOLS_DIR, "../codes_dataset"))

# Load code tables at module import
CODE_TABLES = {
    'items': pd.read_csv(os.path.join(CODES_DIR, "d_items.csv")),
    'diagnoses': pd.read_csv(os.path.join(CODES_DIR, "d_icd_diagnoses.csv")),
    'procedures': pd.read_csv(os.path.join(CODES_DIR, "d_icd_procedures.csv")),
    'labitems': pd.read_csv(os.path.join(CODES_DIR, "d_labitems.csv"))
}

@cached_tool
def lookup_medical_code(search_term: str, code_type: str = "items") -> dict:
    """Look up medical codes by search term."""
    try:
        if code_type not in CODE_TABLES:
            return {"error": f"Invalid code_type. Must be one of: {list(CODE_TABLES.keys())}"}

        df = CODE_TABLES[code_type]

        # Search in label column (or long_title for ICD codes)
        search_col = 'label' if code_type in ['items', 'labitems'] else 'long_title'
        results = df[df[search_col].str.contains(search_term, case=False, na=False)]

        # Convert to dict format
        return {
            "Codes": results.to_dict('records'),
        }
    except Exception as e:
        return {"error": str(e)}


# Register tool
tool_registry.register_tool("lookup_medical_code", lookup_medical_code, {
    "type": "function",
    "function": {
        "name": "lookup_medical_code",
        "description": "Look up medical codes (vital signs, labs, diagnoses, procedures) by searching for a term",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Term to search for (e.g., 'respiratory rate', 'pneumonia', 'blood pressure')"
                },
                "code_type": {
                    "type": "string",
                    "description": "Type of code to search: 'items' (vitals/observations), 'labitems' (labs), 'diagnoses' (ICD diagnoses), 'procedures' (ICD procedures)",
                    "enum": ["items", "labitems", "diagnoses", "procedures"]
                }
            },
            "required": ["code_type", "search_term"]
        }
    }
})