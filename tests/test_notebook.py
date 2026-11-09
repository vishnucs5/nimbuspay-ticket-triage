"""Unit and structural tests for the Google Colab notebook (notebook.ipynb).

Verifies notebook format compliance, cell structure, Python syntax across all code cells,
presence of pre-computed execution outputs, and comprehensive coverage of Assignment Parts A-E.
"""

import ast
import json
import re
from pathlib import Path
import pytest

NOTEBOOK_PATH = Path("notebook.ipynb")


@pytest.fixture(scope="module")
def notebook_data():
    """Load notebook JSON."""
    assert NOTEBOOK_PATH.exists(), f"Notebook file {NOTEBOOK_PATH} does not exist"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_notebook_file_exists_and_is_valid_json():
    """Verify notebook exists, is valid JSON, and adheres to nbformat v4."""
    assert NOTEBOOK_PATH.exists(), "notebook.ipynb must exist at project root"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    assert "cells" in nb, "Notebook must contain 'cells'"
    assert "metadata" in nb, "Notebook must contain 'metadata'"
    assert nb.get("nbformat") == 4, "Notebook nbformat must be 4"
    assert nb.get("nbformat_minor", 0) >= 2, "Notebook nbformat_minor must be >= 2"


def test_notebook_structure_and_cell_counts(notebook_data):
    """Verify the notebook contains a comprehensive sequence of markdown and code cells."""
    cells = notebook_data["cells"]
    assert len(cells) >= 18, f"Expected at least 18 cells, found {len(cells)}"
    
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    
    assert len(markdown_cells) >= 8, f"Expected at least 8 markdown narrative cells, got {len(markdown_cells)}"
    assert len(code_cells) >= 8, f"Expected at least 8 executable code cells, got {len(code_cells)}"


def test_notebook_all_code_cells_have_valid_python_syntax(notebook_data):
    """Verify that all code cells in the notebook compile with valid Python syntax."""
    cells = notebook_data["cells"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    
    for idx, cell in enumerate(code_cells):
        raw_source = "".join(cell.get("source", []))
        
        # Filter IPython magic and shell commands for Python AST parsing
        lines = []
        for line in raw_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                # Replace with comment or pass so indentation is preserved
                lines.append(f"# {line}")
            else:
                lines.append(line)
        
        cleaned_source = "\n".join(lines)
        try:
            ast.parse(cleaned_source)
        except SyntaxError as e:
            pytest.fail(f"Code cell {idx + 1} contains invalid Python syntax: {e}\nSource:\n{raw_source}")


def test_notebook_all_code_cells_have_outputs(notebook_data):
    """Verify that all code cells have non-empty execution outputs (pre-computed)."""
    cells = notebook_data["cells"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    
    for idx, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        assert len(outputs) > 0, f"Code cell {idx + 1} has no outputs. The assignment requires pre-computed outputs."
        assert cell.get("execution_count") is not None, f"Code cell {idx + 1} has null execution_count"
        
        # Check that outputs contain textual content or data
        has_content = False
        for out in outputs:
            if "text" in out and len(out["text"]) > 0:
                has_content = True
            elif "data" in out and len(out["data"]) > 0:
                has_content = True
        assert has_content, f"Code cell {idx + 1} outputs contain no text or data payload"


def test_notebook_covers_parts_a_through_e(notebook_data):
    """Verify notebook contains dedicated markdown and code coverage for Parts A through E."""
    all_markdown = " ".join("".join(c.get("source", [])) for c in notebook_data["cells"] if c.get("cell_type") == "markdown")
    all_code = " ".join("".join(c.get("source", [])) for c in notebook_data["cells"] if c.get("cell_type") == "code")
    combined = all_markdown + " " + all_code
    
    # 1. Environment & Hardware
    assert "T4" in combined or "GPU" in combined, "Missing GPU / hardware inspection"
    assert "torch.cuda" in combined or "nvidia-smi" in combined, "Missing CUDA inspection"
    
    # 2. Part A: Data Audit & Cleaning
    assert "Part A" in all_markdown, "Missing Part A heading in markdown"
    assert "cleaning_log.csv" in combined, "Missing cleaning_log.csv reference"
    assert "train_clean.jsonl" in combined, "Missing train_clean.jsonl reference"
    assert ("leakage" in combined.lower() or "contamination" in combined.lower()), "Missing leakage prevention discussion"
    
    # 3. Part B: Baselines
    assert "Part B" in all_markdown, "Missing Part B heading in markdown"
    assert "score.py" in combined, "Missing score.py reference"
    assert ("zero-shot" in combined.lower() or "zeroshot" in combined.lower()), "Missing zero-shot baseline"
    assert "rules" in combined.lower(), "Missing rules-in-prompt reference baseline"
    
    # 4. Part C: Fine-Tuning Pipeline
    assert "Part C" in all_markdown, "Missing Part C heading in markdown"
    assert "Qwen/Qwen2.5-1.5B-Instruct" in combined or "Qwen2.5-1.5B" in combined, "Missing base model reference"
    assert "LoRA" in combined or "lora" in combined, "Missing LoRA reference"
    assert "50M" in combined or "50," in combined or "trainable" in combined.lower(), "Missing trainable parameter budget check"
    assert "assistant" in combined, "Missing assistant completion masking token reference"
    
    # 5. Part D: Controlled Experiments
    assert "Part D" in all_markdown, "Missing Part D heading in markdown"
    assert "Exact Match" in combined or "exact" in combined.lower(), "Missing exact match reporting"
    assert "QLoRA" in combined or "qlora" in combined.lower(), "Missing QLoRA experiment"
    assert "VRAM" in combined or "Memory" in combined, "Missing GPU memory tracking"
    
    # 6. Part E: Error Analysis & Slices
    assert "Part E" in all_markdown, "Missing Part E heading in markdown"
    assert "category" in combined.lower(), "Missing category slicing"
    assert "language" in combined.lower(), "Missing language slicing"
    assert ("case" in combined.lower() or "failure" in combined.lower()), "Missing failure case studies"
    assert ("roadmap" in combined.lower() or "week" in combined.lower()), "Missing 1-week future roadmap"
    
    # 7. Test Predictions
    assert "predictions_test.jsonl" in combined, "Missing predictions_test.jsonl reference"
    assert "test_inputs.jsonl" in combined, "Missing test_inputs.jsonl reference"


def test_notebook_execution_outputs_content(notebook_data):
    """Verify that execution outputs contain key metric values from the project."""
    all_output_text = ""
    for c in notebook_data["cells"]:
        if c.get("cell_type") == "code":
            for out in c.get("outputs", []):
                if "text" in out:
                    all_output_text += "".join(out["text"])
                if "data" in out and "text/plain" in out["data"]:
                    all_output_text += "".join(out["data"]["text/plain"])
    
    # Check that outputs mention key results
    assert "Exact Match" in all_output_text or "exact" in all_output_text.lower(), "Outputs should contain exact match metrics"
    assert "1,330" in all_output_text or "1330" in all_output_text, "Outputs should confirm 1,330 clean rows"
    assert "400" in all_output_text, "Outputs should confirm 400 test predictions"
