import os
import torch
import json
import sys
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
os.environ["PYTHONUTF8"] = "1"

model_id = "unsloth/meta-llama-3.1-8b-bnb-4bit"
output_dir = "models/llama_official_style_lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",
    dtype = torch.bfloat16, 
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

def load_and_tokenize(file_path):
    data = []
    with open(file_path, 'rb') as f:
        for line in f:
            try:
                item = json.loads(line.decode('utf-8', errors='ignore'))
                full_text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}{tokenizer.eos_token}"
                tokenized = tokenizer(full_text, truncation=True, max_length=1024, padding=False)
                data.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["input_ids"].copy()
                })
            except: continue
    return Dataset.from_list(data)

train_dataset = load_and_tokenize("data/train.jsonl")

training_args = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-4,
    num_train_epochs = 1,
    save_steps = 500,
    
    logging_strategy = "steps",
    logging_steps = 1, 
    report_to = "none",           
    disable_tqdm = False,
    
    optim = "paged_adamw_8bit",
    bf16 = True,
    push_to_hub = False,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    args = training_args,
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(output_dir)