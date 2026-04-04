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
from data.datasets import get_pair
from collections import Counter
import os
os.system('cls' if os.name == 'nt' else 'clear')

class StylometryScorer:
    def __init__(self, model="en_core_web_md"):
        self.nlp = spacy.load(model)

    def process_corpus(self, texts):
        return list(self.nlp.pipe(texts))
    
    def get_entropy(self, docs):
        total_entropy = 0
        count = len(docs)
        for doc in docs:
            words = [token.lemma_.lower() for token in doc if not token.is_punct]
            if not words: continue
            
            counts = Counter(words)
            t = len(words)
            entropy = -sum((c/t) * math.log(c/t, 2) for c in counts.values())
            total_entropy += entropy
        return total_entropy / count if count > 0 else 0

    def get_rank(self, docs):
        total_rank = 0
        count = len(docs)
        for doc in docs:
            words = [token.lemma_.lower() for token in doc if not token.is_punct]
            if not words: continue

            counts = Counter(words)
            most_common = counts.most_common()
            ranks = {word: i for i, (word, freq) in enumerate(most_common, 1)}
            
            total_rank_sum = sum(ranks[word] * counts[word] for word in words)
            total_rank += (total_rank_sum / len(words))
        return total_rank / count if count > 0 else 0

    def passive_construct(self, docs):
        total_passive = 0
        total_words = 0

        for doc in docs:
            passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
            total_passive += passive_count
            words_count = len([t for t in doc if not t.is_punct])
            total_words += words_count
        
        return (total_passive / total_words) * 1000