from langchain_openai import ChatOpenAI
import streamlit as st


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

MODEL = st.secrets.get(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=MODEL,
    temperature=0,
)