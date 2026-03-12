import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# -----------------------------
# load knowledge
# -----------------------------

with open("knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

questions = [k["question"] for k in knowledge]

# create embeddings for all questions
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=questions
).data

vectors = [e.embedding for e in embeddings]


# -----------------------------
# semantic search
# -----------------------------

def search(question):

    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    scores = cosine_similarity([q_embedding], vectors)[0]

    top_indices = scores.argsort()[-3:][::-1]

    results = [knowledge[i] for i in top_indices]

    # אם אין התאמה טובה
    if scores[top_indices[0]] < 0.6:
        return [{
            "question": "",
            "answer": "אשמח לעזור 🙂 אפשר לפרט קצת יותר על השאלה?"
        }]

    return results


# -----------------------------
# Test endpoint for Callbell
# -----------------------------

@app.get("/webhook")
async def test_webhook():
    return {"status": "ok"}


# -----------------------------
# Main webhook endpoint
# -----------------------------

@app.post("/webhook")
async def webhook(data: dict):

    print("Incoming data:", data)

    # תמיכה בכמה פורמטים אפשריים של Callbell
    user_message = (
        data.get("message", {}).get("text")
        or data.get("text")
        or data.get("message")
        or ""
    )

    print("User message:", user_message)

    results = search(user_message)

    context = ""

    for r in results:
        context += f"""
שאלה: {r['question']}
תשובה: {r['answer']}
"""

    prompt = f"""
אתה נציג שירות של המרכז ללימודי המשך של אוניברסיטת תל אביב.

ענה בצורה טבעית, קצרה וברורה.

ידע מהמאגר:
{context}

שאלת משתמש:
{user_message}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        reply_text = response.choices[0].message.content

    except Exception as e:

        print("OpenAI error:", e)

        reply_text = "מצטער, הייתה תקלה זמנית. אפשר לנסות שוב בעוד רגע."

    print("AI reply:", reply_text)

    return {"reply": reply_text}