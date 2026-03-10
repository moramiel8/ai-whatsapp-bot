import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# load knowledge
with open("knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

questions = [k["question"] for k in knowledge]

# create embeddings for all questions
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=questions
).data

vectors = [e.embedding for e in embeddings]


def search(question):

    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    scores = cosine_similarity([q_embedding], vectors)[0]

    # top 3 results
    top_indices = scores.argsort()[-3:][::-1]

    results = [knowledge[i] for i in top_indices]

    # אם אין התאמה טובה
    if scores[top_indices[0]] < 0.6:
        return [{
            "question": "",
            "answer": "אשמח לעזור 🙂 אפשר לפרט קצת יותר על השאלה?"
        }]

    return results


@app.post("/webhook")
async def webhook(data: dict):

    print(data)

    user_message = data.get("text") or data.get("message") or ""

    results = search(user_message)

    context = ""
    for r in results:
        context += f"""
שאלה: {r['question']}
תשובה: {r['answer']}
"""

    prompt = f"""
אתה נציג שירות של המרכז ללימודי המשך של אוניברסיטת תל אביב.

אם המשתמש מבקש לדבר עם נציג אנושי,
או אם אינך בטוח בתשובה,
החזר JSON בצורה:

{{"reply":"", "handoff": true}}

אם אתה יכול לענות, החזר:

{{"reply":"התשובה שלך", "handoff": false}}

ידע מהמאגר:
{context}

שאלת משתמש:
{user_message}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

reply_text = response.choices[0].message.content

try:
    ai_data = json.loads(reply_text)

    if "reply" not in ai_data:
        ai_data["reply"] = reply_text

    if "handoff" not in ai_data:
        ai_data["handoff"] = False

except:
    ai_data = {
        "reply": reply_text,
        "handoff": False
    }

return ai_data