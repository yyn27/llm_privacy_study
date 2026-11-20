"""Single frozen LoRA specification shared by all three models (M0-M11)."""
from peft import LoraConfig

TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=list(TARGET_MODULES),
    task_type="CAUSAL_LM",
)
