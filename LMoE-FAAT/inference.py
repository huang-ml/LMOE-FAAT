import argparse
import torch
from transformers import AutoTokenizer
from modeling_qwen3_moe_lmoe import Qwen3MoeForCausalLMLoad


def main():
    parser = argparse.ArgumentParser(description="Run inference with LMoE-FAAT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MoE model")
    parser.add_argument("--prompt", type=str, default="What is machine learning?", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
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
