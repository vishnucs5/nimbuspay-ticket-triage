"""Tests for fine-tuning engine and configuration module.

Covers:
- Configuration validation and parameter budget checks (<= 4B total, <= 50M trainable)
- ChatML prompt formatting for Qwen2.5
- Completion loss masking logic (-100 for prompt, token IDs for completion)
- Dry-run training pipeline execution and PEFT adapter serialization
- Support for all 4 controlled experiment ablation configurations
"""

import json
from pathlib import Path
import pytest

from src.config import TrainConfig
from src.train import (
    build_chatml_prompt,
    calculate_lora_trainable_params,
    prepare_dataset,
    run_training,
)


def test_default_config_validates():
    """Default TrainConfig must conform to all assignment constraints."""
    cfg = TrainConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.learning_rate == 2e-4
    assert cfg.mask_prompt is True
    assert cfg.max_seq_length == 512
    assert cfg.per_device_train_batch_size == 4
    assert cfg.gradient_accumulation_steps == 4

    # Validation should pass without exception
    cfg.validate()


def test_parameter_budget_calculation():
    """Verify exact formula calculation of trainable parameters for Qwen2.5-1.5B."""
    # Rank 16 default
    cfg16 = TrainConfig(lora_r=16)
    total_params, trainable_16, pct_16 = calculate_lora_trainable_params(cfg16)
    assert trainable_16 == 18_464_768
    assert trainable_16 <= 50_000_000
    assert total_params <= 4_000_000_000
    assert 1.0 < pct_16 < 1.3

    # Rank 8 ablation
    cfg8 = TrainConfig(lora_r=8)
    _, trainable_8, _ = calculate_lora_trainable_params(cfg8)
    assert trainable_8 == 9_232_384
    assert trainable_8 <= 50_000_000

    # Rank 32 ablation
    cfg32 = TrainConfig(lora_r=32)
    _, trainable_32, _ = calculate_lora_trainable_params(cfg32)
    assert trainable_32 == 36_929_536
    assert trainable_32 <= 50_000_000


def test_parameter_budget_enforcement_exceed_trainable():
    """Config validation must raise ValueError if trainable params exceed 50M."""
    cfg_oversize = TrainConfig(lora_r=64)  # 1,154,048 * 64 = 73,859,072 > 50M
    with pytest.raises(ValueError, match="exceed.*50M"):
        cfg_oversize.validate()


def test_parameter_budget_enforcement_exceed_total_model_size():
    """Config validation must raise ValueError if base model exceeds 4B total params."""
    cfg_large_model = TrainConfig(model_name="Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(ValueError, match="exceed.*4B"):
        cfg_large_model.validate()


def test_build_chatml_prompt():
    """Verify ChatML prompt format matches Qwen2.5 specification."""
    sys_msg = "You are NimbusPay triage."
    user_msg = "Customer ticket text"
    asst_msg = '{"category": "refund_status"}'

    # Full prompt with assistant target for training
    full_prompt = build_chatml_prompt(sys_msg, user_msg, asst_msg)
    expected_full = (
        "<|im_start|>system\n"
        "You are NimbusPay triage.<|im_end|>\n"
        "<|im_start|>user\n"
        "Customer ticket text<|im_end|>\n"
        "<|im_start|>assistant\n"
        '{"category": "refund_status"}<|im_end|>\n'
    )
    assert full_prompt == expected_full

    # Prompt without assistant for test-time inference
    infer_prompt = build_chatml_prompt(sys_msg, user_msg, None)
    expected_infer = (
        "<|im_start|>system\n"
        "You are NimbusPay triage.<|im_end|>\n"
        "<|im_start|>user\n"
        "Customer ticket text<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert infer_prompt == expected_infer


def test_prepare_dataset_completion_masking(tmp_path):
    """Verify completion masking sets prompt tokens to -100 and keeps completion tokens."""
    # Create sample JSONL file
    sample_file = tmp_path / "sample_train.jsonl"
    record = {
        "id": "tr-00001",
        "messages": [
            {"role": "system", "content": "You are NimbusPay triage."},
            {"role": "user", "content": "My payment failed yesterday"},
            {"role": "assistant", "content": '{"category": "payment_failure", "priority": "P2"}'},
        ],
    }
    sample_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    dataset = prepare_dataset(
        data_path=str(sample_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        mask_prompt=True,
    )

    assert len(dataset) == 1
    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]
    attention_mask = item["attention_mask"]

    assert len(input_ids) == len(labels) == len(attention_mask)

    # Prompt portion must be -100
    neg_100_count = sum(1 for x in labels if x == -100)
    assert neg_100_count > 0

    # Non-masked portion must match input_ids
    active_labels = [l for l in labels if l != -100]
    assert len(active_labels) > 0
    active_indices = [i for i, l in enumerate(labels) if l != -100]
    for idx in active_indices:
        assert labels[idx] == input_ids[idx]

    # Active labels decode to the assistant content + end tag
    decoded_active = tokenizer.decode(active_labels)
    assert '{"category": "payment_failure"' in decoded_active


def test_prepare_dataset_unmasked_ablation(tmp_path):
    """When mask_prompt=False, prompt tokens are NOT masked with -100."""
    sample_file = tmp_path / "sample_train.jsonl"
    record = {
        "id": "tr-00001",
        "messages": [
            {"role": "system", "content": "You are NimbusPay triage."},
            {"role": "user", "content": "My payment failed yesterday"},
            {"role": "assistant", "content": '{"category": "payment_failure", "priority": "P2"}'},
        ],
    }
    sample_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    dataset = prepare_dataset(
        data_path=str(sample_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        mask_prompt=False,
    )

    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]

    # All unpadded tokens in labels must match input_ids
    assert labels == input_ids


def test_dry_run_training_execution(tmp_path):
    """Verify run_training execution in dry_run mode produces PEFT adapter artifacts."""
    output_dir = tmp_path / "test_adapter"
    cfg = TrainConfig(
        output_dir=str(output_dir),
        lora_r=16,
        dry_run=True,
    )

    result = run_training(cfg)

    assert "training_minutes" in result
    assert "peak_vram_gb" in result
    assert "trainable_params" in result
    assert result["trainable_params"] == 18_464_768
    assert result["output_dir"] == str(output_dir)

    # Check adapter files exist
    assert (output_dir / "adapter_config.json").exists()
    assert (
        (output_dir / "adapter_model.safetensors").exists()
        or (output_dir / "adapter_model.bin").exists()
    )

    # Verify adapter_config.json has valid PEFT structure
    with open(output_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    assert adapter_cfg.get("r") == 16
    assert adapter_cfg.get("lora_alpha") == 32


def test_ablation_matrix_configurations():
    """Verify all 4 required experiment configurations can be parameterized and validated."""
    # Ablation 1: Data quality (Raw vs Clean)
    cfg_raw = TrainConfig(data_path="candidate_pack/data/train.jsonl")
    cfg_clean = TrainConfig(data_path="data/train_clean.jsonl")
    cfg_raw.validate()
    cfg_clean.validate()

    # Ablation 2: Precision (16-bit vs 4-bit QLoRA)
    cfg_16bit = TrainConfig(use_qlora=False)
    cfg_4bit = TrainConfig(use_qlora=True)
    cfg_16bit.validate()
    cfg_4bit.validate()
    assert cfg_16bit.use_qlora is False
    assert cfg_4bit.use_qlora is True

    # Ablation 3: Capacity (r=8 vs r=16)
    cfg_r8 = TrainConfig(lora_r=8, lora_alpha=16)
    cfg_r16 = TrainConfig(lora_r=16, lora_alpha=32)
    cfg_r8.validate()
    cfg_r16.validate()
    _, t8, _ = calculate_lora_trainable_params(cfg_r8)
    _, t16, _ = calculate_lora_trainable_params(cfg_r16)
    assert t8 == 9_232_384
    assert t16 == 18_464_768

    # Ablation 4: Loss masking (completion-only vs full sequence)
    cfg_masked = TrainConfig(mask_prompt=True)
    cfg_unmasked = TrainConfig(mask_prompt=False)
    cfg_masked.validate()
    cfg_unmasked.validate()
    assert cfg_masked.mask_prompt is True
    assert cfg_unmasked.mask_prompt is False


def test_config_serialization():
    """Verify TrainConfig serializes to dict and reconstructs faithfully."""
    cfg = TrainConfig(lora_r=32, lora_alpha=64, learning_rate=3e-4, use_qlora=True)
    d = cfg.to_dict()
    assert d["lora_r"] == 32
    assert d["use_qlora"] is True
    cfg_rebuilt = TrainConfig.from_dict(d)
    assert cfg_rebuilt == cfg

