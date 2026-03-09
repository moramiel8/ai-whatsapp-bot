import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = FastAPI()
client = OpenAI()

with open("knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

questions = [k["question"] for k in knowledge]

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
    best = scores.argmax()

    return knowledge[best]


@app.post("/webhook")
async def webhook(data: dict):
    user_message = data.get("text", "")

    result = search(user_message)

    prompt = f"""
אתה נציג שירות של המרכז ללימודי המשך של אוניברסיטת תל אביב.

ידע:
שאלה: {result['question']}
תשובה: {result['answer']}

שאלת משתמש:
{user_message}

ענה בעברית בצורה טבעית.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content

    return {"reply": reply}