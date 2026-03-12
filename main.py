import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# -----------------------------
# load website knowledge (RAG)
# -----------------------------

with open("knowledge_base.json","r",encoding="utf-8") as f:
    docs = json.load(f)

vectors = [d["embedding"] for d in docs]


# -----------------------------
# load FAQ from conversations
# -----------------------------

with open("conversations.json","r",encoding="utf-8") as f:
    conversations = json.load(f)


# -----------------------------
# semantic search on website
# -----------------------------

def search(question):

    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    scores = cosine_similarity([q_embedding], vectors)[0]

    top_indices = scores.argsort()[-5:][::-1]

    results = [docs[i] for i in top_indices]

    return results


# -----------------------------
# search FAQ first
# -----------------------------

def search_faq(question):

    question = question.lower()

    for c in conversations:

        if c["question"] in question:
            return c["answer"]

    return None


# -----------------------------
# Test endpoint
# -----------------------------

@app.get("/webhook")
async def test_webhook():
    return {"status": "ok"}


# -----------------------------
# Main webhook
# -----------------------------

@app.post("/webhook")
async def webhook(data: dict):

    print("Incoming data:", data)

    user_message = (
        data.get("message", {}).get("text")
        or data.get("text")
        or data.get("message")
        or ""
    )

    user_message = user_message.strip()

    print("User message:", user_message)

    # -----------------------------
    # handle empty messages
    # -----------------------------

    if not user_message:
        return {"reply": "שלום 🙂 איך אפשר לעזור?"}

    # -----------------------------
    # check FAQ from conversations
    # -----------------------------

    faq_answer = search_faq(user_message)

    if faq_answer:
        return {"reply": faq_answer}

    # -----------------------------
    # search website knowledge
    # -----------------------------

    results = search(user_message)

    context = ""

    for r in results:
        context += f"""
מקור: {r['url']}

{r['text']}
"""

    context = context[:3000]

    system_prompt = """
אתה נציג שירות של המרכז ללימודי המשך של הפקולטה לרפואה באוניברסיטת תל אביב.

ענה בעברית.
ענה בצורה אדיבה, מקצועית וקצרה.
אם יש קישור רלוונטי – ציין אותו.
אם אין מידע ברור – הצע לפנות למרכז.
"""

    prompt = f"""
מידע מהאתר:

{context}

שאלת המשתמש:
{user_message}

ענה על בסיס המידע בלבד.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        reply_text = response.choices[0].message.content

    except Exception as e:

        print("OpenAI error:", e)

        reply_text = "מצטער, הייתה תקלה זמנית. אפשר לנסות שוב בעוד רגע."

    print("AI reply:", reply_text)

    return {"reply": reply_text}