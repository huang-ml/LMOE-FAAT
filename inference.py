import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from src.llama_lmoe.modeling_llama_lmoe import LlamaMoeForCausalLMLoad
from src.qwen_lmoe.modeling_qwen3_lmoe import Qwen3MoeForCausalLMLoad


def _detect_model_type(model_path: str) -> str:
    """Infer model type from config or conversion_config; default to qwen when unsure."""
    conv_cfg_path = os.path.join(model_path, "conversion_config.json")
    if os.path.exists(conv_cfg_path):
        try:
            with open(conv_cfg_path, "r") as f:
                conv_cfg = json.load(f)
            base_name = conv_cfg.get("base_model_name", "").lower()
            if "llama" in base_name:
                return "llama"
            if "qwen" in base_name:
                return "qwen"
        except (OSError, json.JSONDecodeError):
            pass

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            model_type = config.get("model_type") or ""
            arch = "".join(config.get("architectures", []))
            probe = f"{model_type} {arch}".lower()
            if "llama" in probe:
                return "llama"
            if "qwen" in probe:
                return "qwen"
        except (OSError, json.JSONDecodeError):
            pass
    return "qwen"


def _load_tokenizer(model_path: str):
    """Load tokenizer while fixing known regex issue when supported."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run inference with LMoE-FAAT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MoE model")
    parser.add_argument("--prompt", type=str, default="What is machine learning?", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map")
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "qwen", "llama"],
        help="Model family: qwen or llama; auto tries to infer from config",
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")

    model_type = _detect_model_type(args.model_path) if args.model_type == "auto" else args.model_type
    tokenizer = _load_tokenizer(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "llama":
        model = LlamaMoeForCausalLMLoad(args.model_path, device_map=args.device_map)
    else:
        model = Qwen3MoeForCausalLMLoad(args.model_path, device_map=args.device_map)
    
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating response...\n")
    
    # Prepare input
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
