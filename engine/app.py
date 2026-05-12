
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import time

BASE_MODEL = "unsloth/meta-llama-3.1-8b-bnb-4bit"
ADAPTER_PATH = "checkpoints/checkpoint-500"
MAX_NEW_TOKENS = 3096

st.set_page_config(page_title="OSA Chat", page_icon="🐝", layout="wide",)

st.title("Official Style Assistant")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.config.use_cache = True 
    model.eval()
    return model, tokenizer

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

st.success("Model loaded!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Write your text...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        clean_prompt = prompt.strip()
        has_paragraphs = "\n" in clean_prompt

        if has_paragraphs:
            st.info("ℹ️ Режим сохранения структуры: обработка по абзацам.")
            segments = [p.strip() for p in clean_prompt.split('\n') if p.strip()]
        else:
            st.warning("⚠️ Режим суммаризации: обработка сплошным текстом.")
            segments = [clean_prompt]

        for i, segment in enumerate(segments):
            current_full_prompt = (
                f"### Instruction:\nTransform to official style\n\n"
                f"### Input:\n{segment}\n\n### Response:\n"
            )
            
            inputs = tokenizer(current_full_prompt, return_tensors="pt").to(model.device)
            
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024 if has_paragraphs else 2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            segment_text = ""
            for new_text in streamer:
                for char in new_text:
                    segment_text += char
                    message_placeholder.markdown(full_response + segment_text + "▌")
                    time.sleep(0.01)
            
            if has_paragraphs:
                full_response += segment_text + "\n\n"
            else:
                full_response = segment_text

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})