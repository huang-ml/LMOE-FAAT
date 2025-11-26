import argparse
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets, Dataset
from src.qwen_lmoe.modeling_qwen3_lmoe import Qwen3MoeForCausalLMLoad


class MoETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        current_step = self.state.global_step
        total_steps = self.state.max_steps
        
        inputs['current_step'] = current_step
        inputs['total_steps'] = total_steps
        
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def transform_commonsense(sample):
    """Transform CommonsenseQA and OpenBookQA format"""
    q = sample.get("question") or sample.get("question_stem") or ""
    return {
        "query": f"{q} {sample['choices']['text']}",
        "response": sample["answerKey"],
    }


def load_and_prepare_datasets(num_samples=10000):
    """Load and prepare all training datasets"""
    print("Loading datasets...")
    
    # Math dataset
    math_train = load_dataset("meta-math/MetaMathQA", split="train").select(range(num_samples))
    math_val = load_dataset("meta-math/MetaMathQA", split="train").select(range(num_samples, num_samples + 1000))
    math_train = math_train.remove_columns(["original_question", "type"])
    math_val = math_val.remove_columns(["original_question", "type"])
    
    # Code dataset
    code_train = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train").select(range(num_samples))
    code_val = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train").select(range(num_samples, num_samples + 1000))
    code_train = code_train.rename_column("instruction", "query")
    code_val = code_val.rename_column("instruction", "query")
    
    # OpenBookQA dataset
    openbookqa_train = load_dataset("allenai/openbookqa", split="train")
    openbookqa_val = load_dataset("allenai/openbookqa", split="validation")
    openbookqa_train = openbookqa_train.map(transform_commonsense, remove_columns=openbookqa_train.column_names)
    openbookqa_val = openbookqa_val.map(transform_commonsense, remove_columns=openbookqa_val.column_names)
    
    # CommonsenseQA dataset
    commonsense_qa_train = load_dataset("tau/commonsense_qa", split="train")
    commonsense_qa_val = load_dataset("tau/commonsense_qa", split="validation")
    commonsense_qa_train = commonsense_qa_train.map(transform_commonsense, remove_columns=commonsense_qa_train.column_names)
    commonsense_qa_val = commonsense_qa_val.map(transform_commonsense, remove_columns=commonsense_qa_val.column_names)
    
    # Medical dataset
    medical_train = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'en', split="train").select(range(num_samples))
    medical_val = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'en', split="train").select(range(num_samples, num_samples + 1000))
    medical_train = medical_train.remove_columns(["Complex_CoT"])
    medical_train = medical_train.rename_column("Question", "query")
    medical_train = medical_train.rename_column("Response", "response")
    medical_val = medical_val.remove_columns(["Complex_CoT"])
    medical_val = medical_val.rename_column("Question", "query")
    medical_val = medical_val.rename_column("Response", "response")
    
    # Concatenate all datasets
    print("Concatenating datasets...")
    train_dataset = concatenate_datasets([
        commonsense_qa_train,
        openbookqa_train,
        math_train,
        code_train,
        medical_train
    ])
    
    val_dataset = concatenate_datasets([
        commonsense_qa_val,
        openbookqa_val,
        math_val,
        code_val,
        medical_val
    ])
    
    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)
    
    train_dataset = Dataset.from_list(train_dataset)
    val_dataset = Dataset.from_list(val_dataset)
    
    return train_dataset, val_dataset


def tokenize_dataset(dataset, tokenizer, max_length=1024):
    """Tokenize dataset with chat template"""
    def format_chat_template(example):
        messages = [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]},
        ]
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            enable_thinking=False,
            add_generation_prompt=False,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        return {"input_ids": tokenized_chat.squeeze()}
    
    return dataset.map(format_chat_template)


def freeze_base_model(model):
    """Freeze base model parameters, only train experts and gates"""
    for p in model.parameters():
        p.requires_grad = False
    
    for name, module in model.named_modules():
        if "experts" in name.lower() or name.endswith("gate"):
            for p in module.parameters():
                p.requires_grad = True


def main():
    parser = argparse.ArgumentParser(description="Train LMoE-FAAT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MoE model")
    parser.add_argument("--output_dir", type=str, default="./multitask-full", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples per dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading")
    
    args = parser.parse_args()
    
    print("Loading tokenizer and model...")
    # Infer base model from the MoE model path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen3MoeForCausalLMLoad(args.model_path, device_map=args.device_map)
    model = model.to(torch.bfloat16)
    
    # Set up tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load and prepare datasets
    train_dataset, val_dataset = load_and_prepare_datasets(args.num_samples)
    
    print("Tokenizing datasets...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, args.max_length)
    
    # Freeze base model
    print("Freezing base model parameters...")
    freeze_base_model(model)
    
    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Set up training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        bf16=True,
        save_strategy="epoch",
        logging_steps=100,
        torch_empty_cache_steps=100,
    )
    
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
