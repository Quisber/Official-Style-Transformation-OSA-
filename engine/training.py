# файл для дообучения
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

model_id = "unsloth/meta-llama-3.1-8b-bnb-4bit"
output_dir = "models/llama_official_style_lora"
max_seq_length = 1024

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16 
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = "auto",
    torch_dtype = torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}{tokenizer.eos_token}"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset(
    "json", 
    data_files = {"train": "data/train.jsonl", "test": "data/test.jsonl"},
    encoding = "utf-8"
)
dataset = dataset.map(formatting_prompts_func, batched=True)

training_args = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 1, 
    gradient_accumulation_steps = 8,
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    max_steps = 100,      
    save_strategy = "steps",
    save_steps = 50,
    logging_steps = 10,
    optim = "paged_adamw_8bit",  
    bf16 = True,     
    remove_unused_columns = False,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)