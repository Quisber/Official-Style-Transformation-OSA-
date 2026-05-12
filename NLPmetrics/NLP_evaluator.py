import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from bert_score import score
from tqdm import tqdm 

base_model_name = "unsloth/meta-llama-3.1-8b-bnb-4bit"
adapter_path = "checkpoints/checkpoint-500"
test_file = "data/test.jsonl"
batch_size = 4 
sample_size = 50 

all_data = []
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        all_data.append(json.loads(line))

random.seed(42)
if len(all_data) > sample_size:
    test_sample = random.sample(all_data, sample_size)
else:
    test_sample = all_data

inputs_list = [d['input'] for d in test_sample]
references = [d['output'] for d in test_sample]

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

candidates = []

print(f"Запуск генерации ({len(inputs_list)} примеров)...")

for i in tqdm(range(0, len(inputs_list), batch_size)):
    current_batch = inputs_list[i : i + batch_size]

    batch_prompts = [
        f"###Instruction:\nTransform\n\n###Input:\n{text}\n\n###Response:\n" 
        for text in current_batch
    ]
    
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to("cuda")
    
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False 
        )
    
    batch_gen = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    candidates.extend([text.strip() for text in batch_gen])

P, R, F1 = score(
    candidates, 
    references, 
    model_type="distilbert-base-uncased", 
    lang="en", 
    device="cuda",
    verbose=False
)

print(f"Средний Precision: {P.mean().item():.4f}")
print(f"Средний Recall:    {R.mean().item():.4f}")
print(f"Средний F1-Score:  {F1.mean().item():.4f}")