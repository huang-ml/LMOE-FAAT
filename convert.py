import argparse
from transformers import AutoTokenizer
from src.llama_lmoe.modeling_llama_lmoe import LlamaMoeForCausalLMConvert, save_moe_model as save_llama_moe_model
from src.qwen_lmoe.modeling_qwen3_lmoe import Qwen3MoeForCausalLMConvert, save_moe_model

def main():
    parser = argparse.ArgumentParser(description="Convert a base Qwen model to MoE")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for MoE model")
    parser.add_argument("--use_attention_experts", action="store_true", help="Add experts to attention layers")
    parser.add_argument("--use_ffn_experts", action="store_true", default=True, help="Add experts to FFN layers")
    parser.add_argument("--num_attention_experts", type=int, default=4, help="Number of attention experts")
    parser.add_argument("--num_attention_experts_per_token", type=int, default=2, help="Attention experts per token")
    parser.add_argument("--num_ffn_experts", type=int, default=16, help="Number of FFN experts")
    parser.add_argument("--num_ffn_experts_per_token", type=int, default=8, help="FFN experts per token")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_aux_heads", action="store_true", help="Use auxiliary heads for FAAT")
    parser.add_argument("--num_aux_heads", type=int, default=4, help="Number of auxiliary heads")
    parser.add_argument("--aux_loss_alpha", type=float, default=0.8, help="Auxiliary loss alpha")
    
    args = parser.parse_args()
    
    print(f"Converting {args.model_name} to MoE model...")
    print(f"Configuration:")
    print(f"  - Attention experts: {args.use_attention_experts}")
    print(f"  - FFN experts: {args.use_ffn_experts}")
    print(f"  - Num FFN experts: {args.num_ffn_experts}")
    print(f"  - FFN experts per token: {args.num_ffn_experts_per_token}")
    print(f"  - LoRA rank: {args.rank}, alpha: {args.alpha}")
    print(f"  - Use auxiliary heads: {args.use_aux_heads}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Convert model
    moe_model = Qwen3MoeForCausalLMConvert(
        args.model_name,
        use_attention_experts=args.use_attention_experts,
        use_ffn_experts=args.use_ffn_experts,
        num_attention_experts=args.num_attention_experts,
        num_attention_experts_per_token=args.num_attention_experts_per_token,
        num_ffn_experts=args.num_ffn_experts,
        num_ffn_experts_per_token=args.num_ffn_experts_per_token,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        use_aux_heads=args.use_aux_heads,
        num_aux_heads=args.num_aux_heads,
        aux_loss_alpha=args.aux_loss_alpha,
    )
    
    # Save model
    print(f"Saving MoE model to {args.output_dir}...")
    save_moe_model(moe_model, args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Conversion complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
