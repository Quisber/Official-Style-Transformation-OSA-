import sys
import os
import time
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from peft import PeftModel

from stylemetrics.stylemetry_evaluator import StylometryScorer

BASE_MODEL = "unsloth/meta-llama-3.1-8b-bnb-4bit"
ADAPTER_PATH = "checkpoints/checkpoint-500"

st.set_page_config(
    page_title="Official Style Assistant",
    layout="wide"
)

st.title("Official Style Adapter")
st.caption("Система автоматической стилистической адаптации и стилеметрического анализа")

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",
        dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.config.use_cache = True
    model.eval()

    return model, tokenizer


@st.cache_resource
def load_scorer():
    return StylometryScorer()


with st.spinner("Загрузка модели..."):
    model, tokenizer = load_model()
    scorer = load_scorer()

if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

tab1, tab2 = st.tabs([
    "Стилистическая адаптация",
    "Стилеметрический анализ"
])

with tab1:

    st.header("Генерация текста")

    input_text = st.text_area(
        "Исходный текст",
        value=st.session_state.source_text,
        height=350,
        placeholder="Вставьте текст..."
    )

    col1, col2 = st.columns([1, 5])

    generate_btn = col1.button("Генерация")
    clear_btn = col2.button("Очистить")

    if clear_btn:
        st.session_state.source_text = ""
        st.session_state.generated_text = ""
        st.rerun()

    if generate_btn:

        st.session_state.source_text = input_text

        clean_prompt = input_text.strip()

        has_paragraphs = "\n" in clean_prompt

        st.header("Трансформированный текст")
        if has_paragraphs:
            st.toast("Текст будет адаптирован с учетом текущей структуры")
            segments = [
                p.strip()
                for p in clean_prompt.split('\n')
                if p.strip()
            ]
        else:
            st.toast("Текст будет суммаризирован. Для сохранения структуры - разделите абзацы")
            segments = [clean_prompt]

        message_placeholder = st.empty()

        full_response = ""

        for i, segment in enumerate(segments):

            current_full_prompt = (
                f"### Instruction:\nTransform\n\n"
                f"### Input:\n{segment}\n\n"
                f"### Response:\n"
            )

            inputs = tokenizer(
                current_full_prompt,
                return_tensors="pt"
            ).to(model.device)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024 if has_paragraphs else 2048,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            thread = Thread(
                target=model.generate,
                kwargs=generation_kwargs
            )

            thread.start()

            segment_text = ""

            for new_text in streamer:

                for char in new_text:
                    segment_text += char
                    message_placeholder.markdown(
                        full_response + segment_text + "▌"
                    )
                    time.sleep(0.04)

            if has_paragraphs:
                full_response += segment_text + "\n\n"
            else:
                full_response = segment_text
            
        message_placeholder.markdown(full_response)

        st.session_state.generated_text = full_response

with tab2:

    st.header("Стилеметрический анализ")

    col1, col2 = st.columns(2)

    with col1:

        source_text = st.text_area(
            "Исходный текст",
            value=st.session_state.source_text,
            height=300
        )

    with col2:

        target_text = st.text_area(
            "Трансформированный текст",
            value=st.session_state.generated_text,
            height=300
        )

    analyze_btn = st.button("Запуск анализа")

    if analyze_btn:

        if not source_text or not target_text:

            st.error("Заполните оба текстовых поля")

        else:

            ready_sources = scorer.process_corpus([source_text])
            ready_targets = scorer.process_corpus([target_text])

            ent_s = scorer.get_entropy(ready_sources)
            rank_s = scorer.get_rank(ready_sources)
            pass_s = scorer.passive_construct(ready_sources)
            syntax_s = scorer.get_syntax_tree(ready_sources)
            nomin_s = scorer.get_nominality(ready_sources)
            subj_s, intense_s = scorer.get_expressiveness(ready_sources)

            ent_t = scorer.get_entropy(ready_targets)
            rank_t = scorer.get_rank(ready_targets)
            pass_t = scorer.passive_construct(ready_targets)
            syntax_t = scorer.get_syntax_tree(ready_targets)
            nomin_t = scorer.get_nominality(ready_targets)
            subj_t, intense_t = scorer.get_expressiveness(ready_targets)

            st.subheader("Сравнение стилеметрических параметров")

            h1, h2, h3, h4 = st.columns([4, 2, 2, 1.5])

            h1.markdown("#### **Параметр**")
            h2.markdown("#### **Исходный текст**")
            h3.markdown("#### **Трансформированный текст**")
            h4.markdown("#### **Отклонение (%)**")

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

                c4.metric(
                    "",
                    "",
                    delta=f"{diff:.1f}%",
                    label_visibility="collapsed"
                )

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

            st.divider()