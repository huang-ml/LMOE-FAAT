# LMoE-FAAT: Lightweight Mixture of Experts with Future-Aware Auxiliary Training

A lightweight implementation of Mixture of Experts (MoE) for language models using LoRA-based experts, combined with Future-Aware Auxiliary Training (FAAT) for improved training efficiency.

## Features

- 🚀 **LoRA-based MoE**: Efficient expert implementation using Low-Rank Adaptation
- 🔮 **Future-Aware Auxiliary Training**: Multi-step prediction for better convergence
- 🎯 **Flexible Architecture**: Support for both attention and FFN experts
- 💾 **Memory Efficient**: Multi-GPU support with balanced distribution
- 🔧 **Easy Conversion**: Convert any Qwen model to MoE with simple commands

You can find our checkpoint [here](https://huggingface.co/datasets/huyhoangvbck/tree/main)

## Installation

```bash
# Install dependencies
pip install torch transformers datasets peft evaluate

# Or using uv (faster)
pip install uv
uv pip install torch transformers datasets peft evaluate
```

## Quick Start

model_name = [meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B]

### 1. Convert a Base Model to MoE

Convert a pretrained Qwen model to MoE architecture:

```bash
python convert.py \
    --model_name Qwen/Qwen3-0.6B \
    --output_dir ./qwen3-0.6b-moe \
    --use_ffn_experts \
    --num_ffn_experts 16 \
    --num_ffn_experts_per_token 8 \
    --rank 32 \
    --alpha 64 \
    --dropout 0.05 \
    --use_aux_heads \
    --num_aux_heads 4
```

**Parameters:**
- `--model_name`: Base Qwen model (e.g., Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B)
- `--output_dir`: Directory to save the converted MoE model
- `--use_attention_experts`: Add experts to attention layers (optional)
- `--use_ffn_experts`: Add experts to FFN layers (default: True)
- `--num_ffn_experts`: Number of FFN experts (default: 16)
- `--num_ffn_experts_per_token`: Active experts per token (default: 8)
- `--rank`: LoRA rank (default: 32)
- `--alpha`: LoRA alpha parameter (default: 64)
- `--dropout`: LoRA dropout rate (default: 0.05)
- `--use_aux_heads`: Enable Future-Aware Auxiliary Training
- `--num_aux_heads`: Number of auxiliary prediction heads (default: 4)

### 2. Train the MoE Model

Train on multiple tasks (math, code, commonsense QA, medical):

```bash
python train.py \
    --model_path ./qwen3-0.6b-moe \
    --output_dir ./multitask-full \
    --num_samples 10000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --num_epochs 6 \
    --max_length 1024 \
    --device_map auto
```
### 3. Inference the MoE model

```bash
python inference.py \
    --model_path ./qwen3-0.6b-moe \
    --prompt "What is machine learning?" \
    --max_length 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --device_map auto
```

**Parameters:**
- `--model_path`: Path to the converted MoE model
- `--output_dir`: Directory to save training outputs
- `--num_samples`: Number of samples per dataset (default: 10000)
- `--batch_size`: Per-device batch size (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 16)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--num_epochs`: Number of training epochs (default: 6)
- `--max_length`: Maximum sequence length (default: 1024)
- `--device_map`: Device placement strategy (auto/cpu/cuda:0)

### 3. Use in Python Code

```python
from transformers import AutoTokenizer
from LMoE-FAAT.modeling_qwen3_moe_final import Qwen3MoeForCausalLMLoad

# Load model
tokenizer = AutoTokenizer.from_pretrained("./qwen3-0.6b-moe")
model = Qwen3MoeForCausalLMLoad("./qwen3-0.6b-moe", device_map="auto")

# Generate text
inputs = tokenizer("What is machine learning?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Architecture Details

### LoRA-based Experts

Each expert uses Low-Rank Adaptation (LoRA) to efficiently add capacity:
- Base model parameters remain frozen
- Experts add task-specific adaptations via low-rank matrices
- Gating network routes tokens to top-k experts

### Future-Aware Auxiliary Training (FAAT)

FAAT improves training by:
1. **Multi-step Prediction**: Auxiliary heads predict future tokens (t+2, t+3, t+4, ...)
2. **Future Information Fusion**: Project predictions back to enhance current representation
3. **Adaptive Gating**: Learn when to use future information
4. **Decaying Loss Weight**: β = 0.5 × (1 + cos(π × progress)) gradually reduces auxiliary loss

### Multi-GPU Support

The code automatically distributes model layers across GPUs:
- Balanced layer distribution for optimal memory usage
- Auxiliary heads distributed across GPUs
- Efficient cross-device tensor management

## Training Datasets

The default training uses five diverse datasets:

1. **MetaMathQA**: Mathematical reasoning
2. **Magicoder-Evol-Instruct**: Code generation
3. **OpenBookQA**: Commonsense reasoning
4. **CommonsenseQA**: Question answering
5. **Medical-O1-Reasoning**: Medical domain

## Model Configuration

### Small Configuration (0.6B)
```bash
--num_ffn_experts 16 \
--num_ffn_experts_per_token 8 \
--rank 32 \
--alpha 64
```

### Medium Configuration (1.7B)
```bash
--num_ffn_experts 16 \
--num_ffn_experts_per_token 8 \
--rank 32 \
--alpha 64
```

### With Attention Experts
```bash
--use_attention_experts \
--num_attention_experts 4 \
--num_attention_experts_per_token 2
```

## Performance Tips

1. **Multi-GPU Training**: Use `--device_map auto` for automatic GPU distribution
2. **Memory Optimization**: Reduce `--batch_size` or increase `--gradient_accumulation_steps`
3. **Gradient Checkpointing**: Enable with `model.gradient_checkpointing_enable()`
4. **Mixed Precision**: Training uses bf16 by default for efficiency

## File Structure

```
d:\LMOE-FAAT\
├── README.md                                    # This file
├── 
│   ├── modeling_qwen3_moe_final.py             # Main MoE implementation
│   ├── configuration_qwen3.py                   # Model configuration
│   ├── convert.py                               # Conversion script
│   ├── train.py                                 # Training script
│   ├── main.ipynb                              # Training notebook (legacy)
│   └── convert_to_moe.ipynb                    # Conversion notebook (legacy)
```

## Citation

If you use this code, please cite:

```bibtex
@software{lmoe_faat_2024,
  title={LMoE-FAAT: Lightweight Mixture of Experts with Future-Aware Auxiliary Training},
  author={Hoang Nguyen Huy, Ha Viet Nguyen, Linh Nguyen Thi Thuy},
  year={2025},
  url={https://github.com/hoanghelloworld/LMOE-FAAT}
}
```

## License

This project is licensed under the MIT License.

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` or `--max_length`
- Increase `--gradient_accumulation_steps`
- Use fewer experts or smaller rank

### Multi-GPU Issues
- Ensure CUDA is properly installed
- Check GPU memory with `nvidia-smi`
- Try `--device_map cpu` for testing

### Training Instability
- Lower `--learning_rate`
- Increase `--warmup_ratio` in training args
- Reduce number of auxiliary heads

## Contact

For questions or issues, please open an issue on GitHub.
