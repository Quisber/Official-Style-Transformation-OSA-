# будет создан вспомогательный метод оценки валидности выходных данных
# планируются измерения по показателям BERTScore
from bert_score import score
import pandas as pd

files = {
    "Simple Prompt": "NLPmetrics/Simple_based.txt",
    "Prompt with examples": "NLPmetrics/Prompt_based.txt",
    "Fine-tuning (OSA)": "NLPmetrics/OSA.txt"
}

with open("NLPmetrics/Original.txt", "r", encoding="utf-8") as f:
    reference = [f.read()]

results = []

for label, filename in files.items():
    with open(filename, "r", encoding="utf-8") as f:
        candidate = [f.read()]
    
    P, R, F1 = score(candidate, reference, lang="en", verbose=False)
    
    results.append({"Model": label, "F1-Score": F1.item()})

df = pd.DataFrame(results)
print(df.to_string(index=False))