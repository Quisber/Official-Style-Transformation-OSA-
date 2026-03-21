# файл для работы с итоговым выводом и демонстрацией всех метрик
import streamlit as st
from Stylometry_score import StylometryScorer # ваш класс
from datasets import get_pair    # ваша функция чтения

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

        col1, col2, col3 = st.columns(3)
        col1.metric("Публ.энтропия", f"{ent_s:.2f}")
        col2.metric("Оффиц.энтропия", f"{ent_t:.2f}")
        col3.metric("Отклонение", f"{diff:.1f}%", delta=f"{diff:.1f}%")

        st.subheader("Сравнение текстов")
        st.text_area("Оригинал", source, height=150)
        st.text_area("Измененная версия", target, height=150)
    else:
        st.error("ID не найден")