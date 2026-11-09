"""Unit tests for controlled experiments runner and comparison matrix (Part D).

Verifies:
1. Definition of all 4 controlled ablation experiments plus best model reference.
2. Strict isolation of independent variables (changing exactly one variable at a time).
3. Integration with candidate_pack/score.py for metric computation.
4. Correct generation and formatting of Part D markdown comparison table.
5. Generation of dev prediction files conforming to schema and scoring standards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.config import TrainConfig
from src.experiments import (
    EXPERIMENT_SPECS,
    Experiment,
    format_comparison_table,
    run_all_experiments,
    run_experiment,
)


def test_experiment_specs_exist_and_complete():
    """Verify all 4 required experiment configurations plus best model exist."""
    required_keys = ["best_model", "exp1_raw", "exp2_qlora", "exp3_rank8", "exp4_fullloss"]
    for key in required_keys:
        assert key in EXPERIMENT_SPECS, f"Missing required experiment key '{key}' in EXPERIMENT_SPECS"
        exp = EXPERIMENT_SPECS[key]
        assert isinstance(exp, Experiment)
        assert exp.name
        assert exp.what_changed
        assert isinstance(exp.config, TrainConfig)
        assert exp.pred_filename.endswith(".jsonl")


def test_experiments_alter_exactly_one_variable():
    """Verify each ablation alters strictly one independent variable relative to best_model."""
    best_cfg = EXPERIMENT_SPECS["best_model"].config.to_dict()

    # Exp 1 (Data Quality Ablation): Only data_path differs
    exp1_cfg = EXPERIMENT_SPECS["exp1_raw"].config.to_dict()
    diff1 = [k for k in best_cfg if best_cfg[k] != exp1_cfg[k]]
    assert diff1 == ["data_path"]
    assert "train.jsonl" in exp1_cfg["data_path"]
    assert "train_clean.jsonl" in best_cfg["data_path"]

    # Exp 2 (Quantization Ablation): Only use_qlora differs
    exp2_cfg = EXPERIMENT_SPECS["exp2_qlora"].config.to_dict()
    diff2 = [k for k in best_cfg if best_cfg[k] != exp2_cfg[k]]
    assert diff2 == ["use_qlora"]
    assert exp2_cfg["use_qlora"] is True
    assert best_cfg["use_qlora"] is False

    # Exp 3 (Capacity Ablation): Only lora_r and lora_alpha differ (proportionate scaling)
    exp3_cfg = EXPERIMENT_SPECS["exp3_rank8"].config.to_dict()
    diff3 = sorted([k for k in best_cfg if best_cfg[k] != exp3_cfg[k]])
    assert diff3 == ["lora_alpha", "lora_r"]
    assert exp3_cfg["lora_r"] == 8
    assert exp3_cfg["lora_alpha"] == 16
    assert best_cfg["lora_r"] == 16
    assert best_cfg["lora_alpha"] == 32

    # Exp 4 (Loss Masking Ablation): Only mask_prompt differs
    exp4_cfg = EXPERIMENT_SPECS["exp4_fullloss"].config.to_dict()
    diff4 = [k for k in best_cfg if best_cfg[k] != exp4_cfg[k]]
    assert diff4 == ["mask_prompt"]
    assert exp4_cfg["mask_prompt"] is False
    assert best_cfg["mask_prompt"] is True


def test_run_experiment_single(tmp_path):
    """Verify single experiment execution generates predictions and metrics."""
    pred_dir = tmp_path / "data"
    result = run_experiment(
        exp_name="best_model",
        output_dir=pred_dir,
        dev_path="candidate_pack/data/dev.jsonl",
    )

    assert result["experiment_name"] == "best_model"
    assert "what_changed" in result
    assert "dev_exact_match" in result
    assert "dev_mean_field_acc" in result
    assert "json_valid" in result
    assert "peak_vram_gb" in result
    assert "training_minutes" in result
    assert "pred_file" in result

    # Check metrics are within valid score range [0.0, 1.0]
    assert 0.0 <= result["dev_exact_match"] <= 1.0
    assert 0.0 <= result["dev_mean_field_acc"] <= 1.0
    assert 0.0 <= result["json_valid"] <= 1.0

    # Check prediction file was written and has 200 lines
    pred_file = Path(result["pred_file"])
    assert pred_file.exists()
    lines = [line.strip() for line in pred_file.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert len(lines) == 200

    # Each line must have 'id' and 'output'
    first_row = json.loads(lines[0])
    assert "id" in first_row
    assert "output" in first_row


def test_markdown_table_formatting():
    """Verify generated comparison table contains all columns required by Part D."""
    mock_results = [
        {
            "experiment_name": "best_model",
            "what_changed": "Cleaned data, 16-bit LoRA, r=16, completion masking",
            "dev_exact_match": 0.800,
            "dev_mean_field_acc": 0.952,
            "peak_vram_gb": 8.8,
            "training_minutes": 18.5,
        },
        {
            "experiment_name": "exp1_raw",
            "what_changed": "Raw training data (uncleaned)",
            "dev_exact_match": 0.510,
            "dev_mean_field_acc": 0.858,
            "peak_vram_gb": 8.9,
            "training_minutes": 20.2,
        },
    ]

    table = format_comparison_table(mock_results)

    # Check required headers from Part D of ASSIGNMENT.md
    assert "What Changed" in table
    assert "Dev Exact Match" in table
    assert "Dev Mean Field Acc" in table
    assert "Peak GPU Memory" in table
    assert "Training Minutes" in table

    # Check formatted values
    assert "80.0%" in table
    assert "95.2%" in table
    assert "8.8 GB" in table
    assert "18.5 min" in table
    assert "51.0%" in table


def test_run_all_experiments_outputs_and_hierarchy(tmp_path):
    """Verify run_all_experiments executes all 5 models and respects expected trade-offs."""
    pred_dir = tmp_path / "data"
    results, table = run_all_experiments(
        output_dir=pred_dir,
        dev_path="candidate_pack/data/dev.jsonl",
    )

    assert len(results) == 5
    assert isinstance(table, str)
    assert len(table) > 100

    res_by_name = {r["experiment_name"]: r for r in results}

    # Best model must achieve the highest exact match on dev
    best_acc = res_by_name["best_model"]["dev_exact_match"]
    assert best_acc > res_by_name["exp1_raw"]["dev_exact_match"]
    assert best_acc > res_by_name["exp4_fullloss"]["dev_exact_match"]

    # Raw training data ablation must show significant exact match drop
    assert res_by_name["exp1_raw"]["dev_exact_match"] < 0.60

    # 4-bit QLoRA must use strictly lower VRAM than 16-bit LoRA
    assert res_by_name["exp2_qlora"]["peak_vram_gb"] < res_by_name["best_model"]["peak_vram_gb"]

    # Check that all 5 prediction files were generated in output_dir
    expected_files = [
        "dev_pred_best.jsonl",
        "dev_pred_exp1_raw.jsonl",
        "dev_pred_exp2_qlora.jsonl",
        "dev_pred_exp3_rank8.jsonl",
        "dev_pred_exp4_fullloss.jsonl",
    ]
    for fn in expected_files:
        p = pred_dir / fn
        assert p.exists(), f"Expected prediction file {fn} not found in {pred_dir}"
        lines = [line.strip() for line in p.read_text(encoding="utf-8").split("\n") if line.strip()]
        assert len(lines) == 200


def test_cli_execution_single_and_all(tmp_path, capsys):
    """Verify CLI interface runs smoothly with --experiment and --all flags."""
    from src.experiments import main

    pred_dir = tmp_path / "cli_data"

    # Test single experiment
    ret = main([
        "--experiment", "exp3_rank8",
        "--output-dir", str(pred_dir),
        "--dev-path", "candidate_pack/data/dev.jsonl",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Exp 3: Capacity" in captured.out
    assert (pred_dir / "dev_pred_exp3_rank8.jsonl").exists()

    # Test running all experiments with JSON output
    ret = main([
        "--all",
        "--output-dir", str(pred_dir),
        "--dev-path", "candidate_pack/data/dev.jsonl",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    json_data = json.loads(captured.out)
    assert len(json_data) == 5


def test_invalid_experiment_name_raises():
    """Verify requesting an undefined experiment raises KeyError."""
    with pytest.raises(KeyError):
        run_experiment("nonexistent_experiment")


def test_missing_dev_file_raises(tmp_path):
    """Verify missing dev file raises FileNotFoundError."""
    missing = tmp_path / "nonexistent_dev.jsonl"
    with pytest.raises(FileNotFoundError):
        run_experiment("best_model", dev_path=missing)

