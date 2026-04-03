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
    st.header("Поле ввода")
    p_idx = st.text_input("Введите ID (например, CNN_0001):")

if p_idx:
    source, target = get_pair(p_idx)
    
    if source:
        scorer = StylometryScorer()
        ent_s = scorer.get_entropy(source)
        ent_t = scorer.get_entropy(target)
        diff = ((ent_t - ent_s) / ent_s) * 100

        rank_s = scorer.get_rank(source)
        rank_t = scorer.get_rank(target)

        with st.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Публ.энтропия", f"{ent_s:.2f}")
            col2.metric("Оффиц.энтропия", f"{ent_t:.2f}")
            col3.metric("Отклонение", f"{diff:.1f}%", delta=f"{diff:.1f}%")
        
        st.divider()

        with st.container():
            col4, col5 = st.columns(2)
            val_s = f"{rank_s:.2f}" if rank_s is not None else "Ошибка"
            val_t = f"{rank_t:.2f}" if rank_t is not None else "Ошибка"
            col4.metric("Ранговое среднее публ.", val_s)
            col5.metric("Ранговое среднее оффиц.", val_t)

        st.subheader("Сравнение текстов")
        st.text_area("Оригинал", source, height=150)
        st.text_area("Измененная версия", target, height=150)
    else:
        st.error("ID не найден")