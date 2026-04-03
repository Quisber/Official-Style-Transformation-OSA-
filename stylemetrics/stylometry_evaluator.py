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
import nltk
from nltk import FreqDist
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
    
    def get_rank(self, text):
        doc = self.nlp(text)
        words = [token.lemma_.lower() for token in doc if not token.is_punct]
        
        counts = Counter(words)
        most_common = counts.most_common()
        
        ranks = {}
        for i, (word, freq) in enumerate(most_common, 1):
            ranks[word] = i
            
        total_rank_sum = sum(ranks[word] * counts[word] for word in words)
        return total_rank_sum / len(words)
    
scorer = StylometryScorer()
