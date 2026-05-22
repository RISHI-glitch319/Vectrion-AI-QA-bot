from groq import Groq
import streamlit as st

groq_api_key = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=groq_api_key
)

def generate_response(query, context, memory_text=""):

    prompt = f"""
You are Vectrion AI,
an advanced enterprise AI document assistant.

Use ONLY the provided document context.

Conversation Memory:
{memory_text}

Document Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content