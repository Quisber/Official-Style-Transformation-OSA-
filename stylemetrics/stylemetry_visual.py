# отдельный файл для визуального представления стилеметрического измерителя
import streamlit as st
from stylemetry_evaluator import StylometryScorer
import streamlit as st
import io

st.title("Стилеметрический анализ")

with st.sidebar:
    st.header("Загрузка данных")
    uploaded_files = st.file_uploader(
        "Выберите 1 или 2 текстовых файла", 
        type=['txt'], 
        accept_multiple_files=True
    )
    
    st.divider()
    btn_start = st.button(label="Запуск анализа")

if btn_start:
    if len(uploaded_files) < 1:
        st.error("Пожалуйста, загрузите хотя бы один файл")
    else:
        scorer = StylometryScorer()
        texts = []
        filenames = []
        for uploaded_file in uploaded_files:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            texts.append(stringio.read())
            filenames.append(uploaded_file.name)

        source_text = texts[0]
        target_text = texts[1] if len(texts) > 1 else ""

        ready_sources = scorer.process_corpus([source_text])
        
        ent_s = scorer.get_entropy(ready_sources)
        rank_s = scorer.get_rank(ready_sources)
        pass_s = scorer.passive_construct(ready_sources)
        syntax_s = scorer.get_syntax_tree(ready_sources)
        nomin_s = scorer.get_nominality(ready_sources)
        subj_s, intense_s = scorer.get_expressiveness(ready_sources)

        st.header(f"Итоговый расчет метрик")
        
        if len(texts) > 1:
            if len(texts) > 1:
                ready_targets = scorer.process_corpus([target_text])
                ent_t = scorer.get_entropy(ready_targets)
                rank_t = scorer.get_rank(ready_targets)
                pass_t = scorer.passive_construct(ready_targets)
                syntax_t = scorer.get_syntax_tree(ready_targets)
                nomin_t = scorer.get_nominality(ready_targets)
                subj_t, intense_t = scorer.get_expressiveness(ready_targets)
                
                st.subheader("Сравнение стилей: " + filenames[0] + " и " + filenames[1])
                st.write("")

                h1, h2, h3, h4 = st.columns([4, 2, 2, 1.5])
                h1.markdown(f"**Параметр**")
                h2.markdown(f"**{filenames[0]}**")
                h3.markdown(f"**{filenames[1]}**")
                h4.markdown("**Отклонение (%)**")
                st.divider()

                def render_row(label, s_val, t_val):
                    c1, c2, c3, c4 = st.columns([4, 2, 2, 1.5])

                    if s_val == 0:
                        diff = 100.0 if t_val > 0 else 0.0
                    else:
                        diff = ((t_val - s_val) / s_val * 100)
                    
                    c1.markdown(f"##### {label}")
                    
                    c2.markdown(f"##### {s_val:.2f}")

                    c3.markdown(f"##### {t_val:.2f}")
                    
                    c4.metric("", "", delta=f"{diff:.1f}%", label_visibility="collapsed")

                metrics = [
                    ("Лексическая вариативность", ent_s, ent_t),
                    ("Лексическая сложность", rank_s, rank_t),
                    ("Синтаксическая сложность", syntax_s, syntax_t),
                    ("Отстраненность", pass_s, pass_t),
                    ("Номинативность", nomin_s, nomin_t),
                    ("Субъективность", subj_s, subj_t),
                    ("Эмоциональность", intense_s, intense_t)
                ]

                for label, s, t in metrics:
                    render_row(label, s, t)
                    st.write("")

        else:
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("Энтропия", f"{ent_s:.2f}")
                col2.metric("Ранговое среднее", f"{rank_s:.2f}")
                col3.metric("Синтаксическая сложность", f"{syntax_s:.2f}")
            st.divider()

            with st.container():
                col4, col5 = st.columns(2)
                col4.metric("Пассив (на 1000 сл.)", f"{pass_s:.2f}")
                col5.metric("Индекс номинальности", f"{nomin_s:.2f}")
            
            with st.container():
                col6, col7 = st.columns(2)
                col6.metric("Объективность текста (от 0 до 1)", f"{subj_s:.2f}")
                col7.metric("Эмоциональность текста", f"{intense_s:.2f}")


        st.divider()
        st.subheader("Тексты файлов")
        c1, c2 = st.columns(2)
        c1.text_area(filenames[0], source_text, height=200)
        if len(texts) > 1:
            c2.text_area(filenames[1], target_text, height=200)
else:
    st.markdown("""
        ### Добро пожаловать! 
        Для начала работы требуется загрузить файлы в боковой панели (до 2-х)
        
        **Доступный функционал:**
        * **Сравнение метрик** (при загрузке 2 файлов)
        * **Профиль текста** (при загрузке 1 файла)
        * **NLP-метрики:** энтропия, синтаксис, пассивный залог и др.
    """)