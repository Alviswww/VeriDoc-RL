from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True, frozen=True)
class AdapterConfig:
    adapter_type: str = "qlora"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


@dataclass(slots=True, frozen=True)
class PrecisionConfig:
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adapter_config_from_mapping(value: Mapping[str, Any] | None) -> AdapterConfig:
    if value is None:
        return AdapterConfig()
    target_modules = value.get("target_modules", DEFAULT_LORA_TARGET_MODULES)
    return AdapterConfig(
        adapter_type=str(value.get("adapter_type", "qlora")),
        r=int(value.get("r", 16)),
        lora_alpha=int(value.get("lora_alpha", 32)),
        lora_dropout=float(value.get("lora_dropout", 0.05)),
        target_modules=_coerce_string_tuple(target_modules),
        load_in_4bit=bool(value.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(value.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=str(value.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(value.get("bnb_4bit_use_double_quant", True)),
    )


def precision_config_from_mapping(value: Mapping[str, Any] | None) -> PrecisionConfig:
    if value is None:
        return PrecisionConfig()
    attn_implementation = value.get("attn_implementation")
    return PrecisionConfig(
        torch_dtype=str(value.get("torch_dtype", "bfloat16")),
        attn_implementation=str(attn_implementation) if attn_implementation is not None else None,
        gradient_checkpointing=bool(value.get("gradient_checkpointing", True)),
    )


def load_tokenizer(model_name_or_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer loading requires the optional training dependencies "
            "(`transformers` and usually `accelerate`)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    *,
    model_name_or_path: str,
    adapter_config: AdapterConfig,
    precision_config: PrecisionConfig,
) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Model loading requires the optional training dependencies "
            "(`transformers`, `accelerate`, and usually `peft`)."
        ) from exc

    model_kwargs = build_model_load_kwargs(
        adapter_config=adapter_config,
        precision_config=precision_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if precision_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if adapter_config.adapter_type == "qlora":
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requires `peft` in the active training environment."
            ) from exc
        model = prepare_model_for_kbit_training(model)

    if adapter_config.adapter_type in {"lora", "qlora"}:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA / QLoRA training requires `peft` in the active training environment."
            ) from exc
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=adapter_config.r,
            lora_alpha=adapter_config.lora_alpha,
            lora_dropout=adapter_config.lora_dropout,
            target_modules=list(adapter_config.target_modules),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    return model


def load_generation_model(
    *,
    model_name_or_path: str,
    adapter_config: AdapterConfig,
    precision_config: PrecisionConfig,
) -> Any:
    model_kwargs = build_model_load_kwargs(
        adapter_config=adapter_config,
        precision_config=precision_config,
    )
    model_kwargs["device_map"] = "auto"
    model_path = Path(model_name_or_path)

    if model_path.exists() and (model_path / "adapter_config.json").exists():
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Loading a PEFT checkpoint for inference requires `peft`."
            ) from exc
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    else:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Model loading requires the optional training dependencies "
                "(`transformers`, `accelerate`, and usually `peft`)."
            ) from exc
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    return model


def build_model_load_kwargs(
    *,
    adapter_config: AdapterConfig,
    precision_config: PrecisionConfig,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    torch_dtype = resolve_torch_dtype(precision_config.torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if precision_config.attn_implementation:
        kwargs["attn_implementation"] = precision_config.attn_implementation
    if adapter_config.adapter_type == "qlora" or adapter_config.load_in_4bit:
        kwargs["quantization_config"] = build_bitsandbytes_config(adapter_config)
    return kwargs


def build_bitsandbytes_config(adapter_config: AdapterConfig) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA / 4-bit loading requires `transformers` with bitsandbytes support."
        ) from exc

    compute_dtype = resolve_torch_dtype(adapter_config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=adapter_config.load_in_4bit,
        bnb_4bit_quant_type=adapter_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=adapter_config.bnb_4bit_use_double_quant,
    )


def render_chat_messages(messages: Sequence[Mapping[str, Any]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        except Exception:
            pass
    return "\n\n".join(
        f"{str(item.get('role', 'user')).upper()}: {str(item.get('content', '')).strip()}"
        for item in messages
    ).strip()


def resolve_torch_dtype(value: str | None) -> Any:
    if value in {None, "", "auto"}:
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Resolving torch dtype requires `torch` in the active training environment."
        ) from exc

    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return tuple(DEFAULT_LORA_TARGET_MODULES)
