import os
import sys
from datetime import datetime
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

    best_score = scores[top_indices[0]]

    results = [docs[i] for i in top_indices]

    return results, best_score

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

# -----------------------------
# log questions
# -----------------------------

def log_question(question):

    log_file = "questions_log.json"

    entry = {
        "question": question,
        "time": datetime.now().isoformat()
    }

    if os.path.exists(log_file):

        with open(log_file,"r",encoding="utf-8") as f:
            data = json.load(f)

    else:
        data = []

    data.append(entry)

    with open(log_file,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def log_unknown_question(question):

    file = "unknown_questions.json"

    entry = {
        "question": question,
        "time": datetime.now().isoformat()
    }

    if os.path.exists(file):

        with open(file,"r",encoding="utf-8") as f:
            data = json.load(f)

    else:
        data = []

    data.append(entry)

    with open(file,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

# -----------------------------
# Webhook
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

    log_question(user_message)

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

    results, score = search(user_message)

    context = ""

    for r in results:
        context += f"""
מקור: {r['url']}

{r['text']}
"""

    context = context[:3000]

    system_prompt = """
אתה עוזר מידע רשמי של המרכז ללימודי המשך של הפקולטה לרפואה באוניברסיטת תל אביב.

מטרתך לספק מידע ללומדים ומתעניינים בתוכניות הלימוד של המרכז.

כללי מענה:
- כתוב בעברית תקנית ובשפה נייטרלית ומקצועית.
- אל תשתמש באימוג'ים או בניסוחים לא פורמליים.
- שמור על תשובות קצרות וברורות (2–3 משפטים).
- התבסס רק על המידע שסופק מהאתר.
- אל תמציא מידע שאינו מופיע במידע שסופק.
- כאשר יש קישור רלוונטי באתר – ציין אותו בסוף התשובה.
- אם השאלה עוסקת ביצירת קשר, הסבר שניתן לפנות למרכז בטלפון 03-6409797/6409229 בשעות הפעילות או דרך עמוד יצירת הקשר באתר.
- אם אין מידע ברור, הפנה לעמוד יצירת הקשר של המרכז.

סגנון תשובה רצוי:

שאלה: איך ניתן ליצור קשר עם המרכז?
תשובה:
ניתן ליצור קשר עם המרכז ללימודי המשך בטלפון בשעות הפעילות או דרך עמוד יצירת הקשר באתר. צוות המרכז ישמח לסייע.
"""

    prompt = f"""
מידע מהאתר:

{context}

שאלת המשתמש:
{user_message}

ענה בהתאם למידע בלבד ובשפה נייטרלית ומקצועית.
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