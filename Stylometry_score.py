#Планируется создать стилистический измеритель, основанный на общих наглядных признаках официально-делового и публицистического стилей
#Также будут применены методы и подходы стилеметрии
#В сумме измеритель будет фиксировать следующие параметры:
# - Информационная энтропия
# - Ранговое среднее 
# - Количество пассивных конструкций
# - Длина, ширина и глубина деревьев зависимостей
# - Эмоциональность
# - Номинальность
import spacy
import math 
from datasets import get_pair
from collections import Counter
import os
os.system('cls' if os.name == 'nt' else 'clear')

# энтропия
class StylometryScorer:
    def __init__(self, model="en_core_web_md"):
        self.nlp = spacy.load(model)

    def get_entropy(self, text):
        doc = self.nlp(text)
        words = [token.lemma_.lower() for token in doc if not token.is_punct]
        counts = Counter(words)
        total = len(words)
        return -sum((c/total) * math.log(c/total, 2) for c in counts.values())
    
scorer = StylometryScorer()

p_idx = input("Введите ID (например, CNN_0001): ")
source, target = get_pair(p_idx)

if source and target:
    res_source = scorer.get_entropy(source)
    res_target = scorer.get_entropy(target)

print(f"Публицистика:{res_source}, \n Официальная: {res_target}, \n Разница: {round((((res_target - res_source)/res_source)*100), 3)}%")