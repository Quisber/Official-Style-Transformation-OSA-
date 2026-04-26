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
        total_sum_ranks = 0
        total_word_count = 0
        default_rank = 50000 

        for doc in docs:
            for token in doc:
                if not token.is_punct and not token.is_space:
                    if token.has_vector:
                        total_sum_ranks += (token.rank + 1)
                    else:
                        total_sum_ranks += default_rank

                    total_word_count += 1
            
        return total_sum_ranks / total_word_count if total_word_count > 0 else 0

    def passive_construct(self, docs):
        total_passive = 0
        total_words = 0

        for doc in docs:
            passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
            total_passive += passive_count
            words_count = len([t for t in doc if not t.is_punct])
            total_words += words_count
        
        return (total_passive / total_words) * 1000
    
    def get_syntax_tree(self, docs):
        total_complexity_sum = 0
        total_sentences_count = 0
        
        for doc in docs:
            for sent in doc.sents:
                length = len([t for t in sent if not t.is_punct])
                width = len(list(sent.root.children))
                
                def get_depth(node):
                    if not list(node.children): return 0
                    return 1 + max(get_depth(child) for child in node.children)
                depth = get_depth(sent.root)
                
                sent_complexity = (length * 0.2) + (width * 0.3) + (depth * 0.5)
                
                total_complexity_sum += sent_complexity
                total_sentences_count += 1
                
        return total_complexity_sum / total_sentences_count if total_sentences_count > 0 else 0