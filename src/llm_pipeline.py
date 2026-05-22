from groq import Groq
from dotenv import load_dotenv
import os

# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()

# =====================================================
# LOAD API KEY
# =====================================================

groq_api_key = os.getenv("GROQ_API_KEY")

# =====================================================
# INITIALIZE GROQ CLIENT
# =====================================================

client = Groq(
    api_key=groq_api_key
)

# =====================================================
# GENERATE RESPONSE
# =====================================================

def generate_response(query, context, memory_text=""):

    prompt = f"""
You are Vectrion AI,
an advanced enterprise AI document assistant.

You must:
1. Answer ONLY using document context.
2. Cite sources naturally.
3. Mention page numbers.
4. Be professional and concise.
5. If answer is unavailable, clearly say so.

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