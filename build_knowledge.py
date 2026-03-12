import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

urls = [

"https://med.tau.ac.il/about-cme",
"https://med.tau.ac.il/cme-vision",
"https://med.tau.ac.il/contact-cme",
"https://med.tau.ac.il/medical-courses",
"https://med.tau.ac.il/oncology",
"https://med.tau.ac.il/urology",
"https://med.tau.ac.il/orthopedics",
"https://med.tau.ac.il/geriatrics",
"https://med.tau.ac.il/radiology",
"https://med.tau.ac.il/psychiatry",
"https://med.tau.ac.il/pediatrics",
"https://med.tau.ac.il/sports-medicine",
"https://med.tau.ac.il/us-pocus",
"https://med.tau.ac.il/familymed-program",
"https://med.tau.ac.il/psychotherapy",

]

def extract_text(url):

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())

    return text


documents = []

for url in tqdm(urls):

    text = extract_text(url)

    chunks = [text[i:i+800] for i in range(0, len(text), 800)]

    for chunk in chunks:

        documents.append({
            "url": url,
            "text": chunk
        })


print("creating embeddings...")

embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=[d["text"] for d in documents]
).data


for i,e in enumerate(embeddings):

    documents[i]["embedding"] = e.embedding


with open("knowledge_base.json","w",encoding="utf-8") as f:
    json.dump(documents,f,ensure_ascii=False)

print("knowledge_base.json created")