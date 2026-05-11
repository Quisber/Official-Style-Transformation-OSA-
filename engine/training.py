import os
import torch
import json
import sys
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import Trainer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
os.environ["PYTHONUTF8"] = "1"

model_id = "unsloth/meta-llama-3.1-8b-bnb-4bit"
output_dir = "models/llama_official_style_lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",
    dtype = torch.bfloat16, 
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.037,
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
                tokenized = tokenizer(full_text, truncation=True, max_length=512, padding="max_length", return_tensors=None )
                
                data.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": list(tokenized["input_ids"]) 
                })
            except Exception as e:
                print(e)
    return Dataset.from_list(data)

train_dataset = load_and_tokenize("data/train.jsonl")

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1.47e-4,
    weight_decay=0.079,
    bf16=True,
    save_strategy="epoch",
    logging_steps=50
)

trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    args = training_args,
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)