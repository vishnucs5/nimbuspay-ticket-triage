"""Training configuration and parameter budgeting for LoRA/QLoRA fine-tuning.

Encapsulates all hyperparameters, hardware constraints, and validation
for the NimbusPay ticket triage model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Assignment-mandated hard constraints
MAX_TOTAL_PARAMS = 4_000_000_000       # strictly <= 4B parameters
MAX_TRAINABLE_PARAMS = 50_000_000     # strictly <= 50M trainable LoRA parameters

# Known open-weight models parameter dictionary for exact zero-latency verification
KNOWN_MODEL_PARAMS: Dict[str, int] = {
    "Qwen/Qwen2.5-0.5B-Instruct": 494_032_768,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_543_714_304,
    "Qwen/Qwen2.5-3B-Instruct": 3_086_329_344,
    "Qwen/Qwen2.5-7B-Instruct": 7_615_616_512,
    "Qwen/Qwen2.5-14B-Instruct": 14_770_231_296,
    "Qwen/Qwen2.5-32B-Instruct": 32_512_000_000,
    "Qwen/Qwen2.5-72B-Instruct": 72_706_000_000,
    "meta-llama/Llama-3.2-1B-Instruct": 1_235_814_400,
    "meta-llama/Llama-3.2-3B-Instruct": 3_212_749_824,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8_030_261_248,
}

# Qwen2.5-1.5B structural parameters
QWEN_1_5B_LAYERS = 28
QWEN_1_5B_HIDDEN_SIZE = 1536
QWEN_1_5B_INTERMEDIATE_SIZE = 8960
QWEN_1_5B_KV_HEADS = 2
QWEN_1_5B_HEAD_DIM = 128  # 1536 // 12 heads = 128


def estimate_base_model_params(model_name: str) -> int:
    """Estimate or lookup base model parameter count without downloading full weights."""
    if model_name in KNOWN_MODEL_PARAMS:
        return KNOWN_MODEL_PARAMS[model_name]

    # Pattern match for models like "...-7B-..." or "...-14B-..."
    match = re.search(r"(\d+(\.\d+)?)B", model_name, re.IGNORECASE)
    if match:
        billions = float(match.group(1))
        return int(billions * 1_000_000_000)

    # Fallback to default Qwen2.5-1.5B parameter count
    return KNOWN_MODEL_PARAMS["Qwen/Qwen2.5-1.5B-Instruct"]


def calculate_lora_trainable_params(cfg: "TrainConfig") -> Tuple[int, int, float]:
    """Calculate exact total parameters, trainable parameters, and trainable ratio.

    For Qwen2.5-1.5B:
    - 28 transformer decoder layers
    - In/Out dimensions for linear projections targeted by LoRA:
      - q_proj: (1536, 1536) -> (1536 + 1536) * r = 3072 * r
      - k_proj: (1536, 256)  -> (1536 + 256) * r  = 1792 * r
      - v_proj: (1536, 256)  -> (1536 + 256) * r  = 1792 * r
      - o_proj: (1536, 1536) -> (1536 + 1536) * r = 3072 * r
      - gate_proj: (1536, 8960) -> (1536 + 8960) * r = 10496 * r
      - up_proj:   (1536, 8960) -> (1536 + 8960) * r = 10496 * r
      - down_proj: (8960, 1536) -> (8960 + 1536) * r = 10496 * r
    Total per layer: 41,216 * r
    Total across 28 layers: 1,154,048 * r

    Returns:
        (total_parameters, trainable_parameters, trainable_percentage)
    """
    base_params = estimate_base_model_params(cfg.model_name)
    r = cfg.lora_r

    # Exact parameter counts per projection module for Qwen2.5-1.5B
    module_weights: Dict[str, int] = {
        "q_proj": (QWEN_1_5B_HIDDEN_SIZE + QWEN_1_5B_HIDDEN_SIZE) * r,
        "k_proj": (QWEN_1_5B_HIDDEN_SIZE + (QWEN_1_5B_KV_HEADS * QWEN_1_5B_HEAD_DIM)) * r,
        "v_proj": (QWEN_1_5B_HIDDEN_SIZE + (QWEN_1_5B_KV_HEADS * QWEN_1_5B_HEAD_DIM)) * r,
        "o_proj": (QWEN_1_5B_HIDDEN_SIZE + QWEN_1_5B_HIDDEN_SIZE) * r,
        "gate_proj": (QWEN_1_5B_HIDDEN_SIZE + QWEN_1_5B_INTERMEDIATE_SIZE) * r,
        "up_proj": (QWEN_1_5B_HIDDEN_SIZE + QWEN_1_5B_INTERMEDIATE_SIZE) * r,
        "down_proj": (QWEN_1_5B_INTERMEDIATE_SIZE + QWEN_1_5B_HIDDEN_SIZE) * r,
    }

    per_layer_trainable = sum(module_weights.get(m, 0) for m in cfg.target_modules)
    trainable_params = per_layer_trainable * QWEN_1_5B_LAYERS
    total_params = base_params + trainable_params
    trainable_pct = (trainable_params / total_params) * 100.0

    return total_params, trainable_params, trainable_pct


@dataclass
class TrainConfig:
    """Hyperparameter configuration for LoRA and QLoRA fine-tuning."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "data/train_clean.jsonl"
    output_dir: str = "adapter"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_qlora: bool = False  # False for 16-bit LoRA; True for 4-bit NF4 QLoRA
    use_rslora: bool = True  # Rank-Stabilized LoRA scaling (alpha / sqrt(r))

    # Optimization & training schedule
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    max_seq_length: int = 512
    mask_prompt: bool = True  # Completion-only loss masking on `<|im_start|>assistant\n`
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Logging and checkpoints
    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 42

    # CI / offline / test harness controls
    dry_run: bool = False
    max_steps: Optional[int] = None

    def validate(self) -> None:
        """Validate all parameters against assignment rules and hardware constraints.

        Raises:
            ValueError: If parameters violate the <= 4B total or <= 50M trainable limits.
        """
        total_params, trainable_params, _ = calculate_lora_trainable_params(self)

        base_params = estimate_base_model_params(self.model_name)
        if base_params > MAX_TOTAL_PARAMS:
            raise ValueError(
                f"Base model '{self.model_name}' has estimated {base_params:,} parameters, "
                f"which exceeds the strict assignment ceiling of 4B ({MAX_TOTAL_PARAMS:,}) parameters."
            )

        if total_params > MAX_TOTAL_PARAMS:
            raise ValueError(
                f"Total model parameters ({total_params:,}) exceed the strict ceiling of 4B "
                f"({MAX_TOTAL_PARAMS:,}) parameters."
            )

        if trainable_params > MAX_TRAINABLE_PARAMS:
            raise ValueError(
                f"Trainable LoRA parameters ({trainable_params:,}) exceed the strict assignment "
                f"budget of 50M ({MAX_TRAINABLE_PARAMS:,}) parameters. "
                f"Consider reducing lora_r (currently {self.lora_r})."
            )

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs}")
        if self.per_device_train_batch_size <= 0:
            raise ValueError(f"batch size must be positive, got {self.per_device_train_batch_size}")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """Instantiate configuration from dictionary."""
        return cls(**data)
