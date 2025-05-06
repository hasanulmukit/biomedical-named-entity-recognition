import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_ner():
    return pipeline("ner", model="outputs", tokenizer="outputs")

st.title("Biomedical NER Demo")
text = st.text_area("Enter abstract or paragraph:")
if text:
    ner = load_ner()
    ents = ner(text)
    for e in ents:
        st.write(f"**{e['entity_group']}**: {e['word']} ({e['score']:.2f})")
