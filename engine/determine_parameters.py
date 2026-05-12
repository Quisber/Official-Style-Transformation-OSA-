import torch
import optuna
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bert_score import score

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output} <|end_of_text|>"
        texts.append(text)
    return { "text":texts }

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
dataset = dataset.shuffle(seed=42)

train_subset = dataset.select(range(150)).map(formatting_prompts_func, batched=True)
val_raw = dataset.select(range(150, 200))
val_data = {
    "src": list(val_raw["input"]), 
    "ref": list(val_raw["output"])
}

def evaluate_model(model, tokenizer, test_cases):
    model.eval()
    generated = []
    
    src_texts = test_cases['src'][:25]
    ref_texts = test_cases['ref'][:25]
    
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 4
    for i in range(0, len(src_texts), batch_size):
        batch_src = src_texts[i : i + batch_size]
        prompts = [f"### Instruction:\nTransform\n\n### Input:\n{s}\n\n### Response:\n" for s in batch_src]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        for output in outputs:
            full_text = tokenizer.decode(output, skip_special_tokens=True)
            resp = full_text.split("### Response:\n")[-1].strip()
            generated.append(resp)
            
    P, R, F1 = score(generated, ref_texts, lang="en", verbose=False)
    return F1.mean().item()

def objective(trial):
    r = trial.suggest_categorical("r", [8, 16, 32, 64])
    alpha_multiplier = trial.suggest_categorical("alpha_multiplier", [1, 2, 4])
    lora_alpha = r * alpha_multiplier
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    
    model_id = "unsloth/meta-llama-3.1-8b-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa" 
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_train = train_subset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            output_dir="outputs_hpo",
            remove_unused_columns=True,
            report_to="none",
            weight_decay=weight_decay
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    
    current_score = evaluate_model(model, tokenizer, val_data)
    
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return current_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)