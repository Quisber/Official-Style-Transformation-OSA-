# отдельный файл для визуального представления стилеметрического измерителя
import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stylemetrics.stylemetry_evaluator import StylometryScorer
from data.datasets import get_pair

st.set_page_config(page_title="Style Analysis", layout="wide")
st.title("Стилеметрический анализ")

with st.sidebar:
    st.header("Кнопка активации")
    btn_start = st.button(label = "Запуск анализа по корпусу")

if btn_start:
    scorer = StylometryScorer()
    sources, targets = get_pair()
    
    ready_sources = scorer.process_corpus(sources)
    ent_s = scorer.get_entropy(ready_sources)
    rank_s = scorer.get_rank(ready_sources)
    pass_s = scorer.passive_construct(ready_sources)

    ready_targets = scorer.process_corpus(targets)
    ent_t = scorer.get_entropy(ready_targets)
    rank_t = scorer.get_rank(ready_targets)
    pass_t = scorer.passive_construct(ready_targets)
    
    if sources:
        diff = ((ent_t - ent_s) / ent_s) * 100

        with st.container(): # отображение энтропии
            col1, col2, col3 = st.columns(3)
            col1.metric("Публ.энтропия", f"{ent_s:.2f}")
            col2.metric("Оффиц.энтропия", f"{ent_t:.2f}")
            col3.metric("Отклонение", f"{diff:.1f}%", delta=f"{diff:.1f}%")
        
        st.divider()

        with st.container(): # отображение рангового среднего
            col4, col5 = st.columns(2)
            val_s = f"{rank_s:.2f}" 
            val_t = f"{rank_t:.2f}" 
            col4.metric("Ранговое среднее публ.", f"{rank_s:.2f}")
            col5.metric("Ранговое среднее оффиц.", f"{rank_t:.2f}")

        st.divider()

        with st.container(): # отображение колличества пассива
            col6, col7 = st.columns(2)
            col6.metric(f"Кол-во пассивных конструкций публ. на 1000 слов", f"{pass_s:.2f}")
            col7.metric(f"Кол-во пассивных конструкций оффиц. на 1000 слов", f"{pass_t:.2f}")

        st.subheader("Сравнение текстов")
        st.text_area("Оригинал", sources, height=150)
        st.text_area("Измененная версия", targets, height=150)