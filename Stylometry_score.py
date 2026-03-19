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
from collections import Counter

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
public = scorer.get_entropy("Republicans are under pressure to give final approval to a deal between President Donald Trump and Senate Democrats that temporarily extends Department of Homeland Security funding for two weeks — alongside a broader, full-year spending deal — so the two parties can negotiate over Democrats’ demands to rein in Immigration and Customs Enforcement tactics.") 
official = scorer.get_entropy("The Republican caucus is currently mandated to provide final ratification for a bilateral agreement reached between the Executive Branch and Senate Democrats, providing for a provisional fourteen-day extension of Department of Homeland Security (DHS) appropriations, concurrent with a comprehensive full-year fiscal expenditure framework.")
print(public, official, "-->", round((((official - public)/public)*100), 2), "%")