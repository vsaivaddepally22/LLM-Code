import re
import math
import requests
import json
import numpy as np
import faiss
from gensim.models import Word2Vec

HARD_CODED_TEXT = """
Levnono is a computer manufacturing company in chanda nagar.\n
Levnono produces laptops, desktops and hardware components.\n
They focus on quality and have a customer support center in Hyderabad.
"""

# Document segmentation (each separate text chunk is a doc)
DOCS = [d.strip() for d in HARD_CODED_TEXT.strip().split("\n") if d.strip()]
EMBED_DIM = 128


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def train_word2vec(docs: list[str], vector_size: int = 100):
    sentences = [tokenize(doc) for doc in docs if doc.strip()]
    if not sentences:
        return None
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=5, min_count=1, workers=1, epochs=100)
    return model


word2vec_model = train_word2vec(DOCS, vector_size=EMBED_DIM)


def embed_text(text: str):
    tokens = tokenize(text)
    if not tokens or word2vec_model is None:
        return [0.0] * EMBED_DIM

    vectors = []
    for token in tokens:
        if token in word2vec_model.wv:
            vectors.append(word2vec_model.wv[token])
    if not vectors:
        return [0.0] * EMBED_DIM

    vec = np.mean(np.vstack(vectors), axis=0)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()


def build_faiss_index(docs: list[str]):
    embeddings = np.array([embed_text(doc) for doc in docs], dtype=np.float32)
    if embeddings.size == 0:
        return None
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def retrieve_most_relevant(question: str, docs: list[str], top_k: int = 1):
    if not question:
        return []
    index = build_faiss_index(docs)
    if index is None:
        return []

    query_emb = np.array([embed_text(question)], dtype=np.float32)
    faiss.normalize_L2(query_emb)

    # Debug: print query embedding (first 8 dims) and each doc embedding similarity
    print("DEBUG query_emb", query_emb[0][:8].tolist())
    for i, doc in enumerate(docs):
        doc_emb = np.array(embed_text(doc), dtype=np.float32)
        faiss.normalize_L2(doc_emb.reshape(1, -1))
        score = float(np.dot(query_emb[0], doc_emb))
        print(f"DEBUG doc[{i}] score={score:.4f} doc='{doc[:60]}'")

    scores, idxs = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if score <= 0:
            continue
        results.append((float(score), docs[int(idx)]))
    return results


from flask import Flask, request, render_template_string

app = Flask(__name__)
app.secret_key = "chatbot-secret"

HTML = """
<!doctype html>
<html><head><title>LLM Chatbot</title></head><body>
<h1>LLM Chatbot</h1>
<form method=post>
  <label for=question>Question:</label><br>
  <input id=question name=question style='width:500px;' value='{{question|default("")}}'><br><br>
  <button type=submit>Ask</button>
</form>
{% if context %}
  <div style='margin-top:1rem;padding:8px;border:1px solid #ddd; background:#fafafa;'>
    <strong>Retrieved Context:</strong> {{context}}
  </div>
{% endif %}
{% if answer %}
  <div style='margin-top:1rem;padding:8px;border:1px solid #444; background:#f2f2ff;'>
    <strong>LLM Response:</strong><br>
    <pre style='white-space:pre-wrap; font-family:inherit;'>{{answer}}</pre>
  </div>
{% endif %}
</body></html>
"""


def get_answer(question: str):
    relevant_docs = retrieve_most_relevant(question, DOCS, top_k=1)
    if not relevant_docs:
        return "No relevant information found.", ""
    score, context = relevant_docs[0]
    prompt = (
        "Use the following context to answer the question. If the context is insufficient, say 'I don't know.'\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemma-3-4b-it:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 250
        })
    )
    if response.status_code == 429:
        return "LLM rate limit exceeded (429). Please wait a few seconds and retry.", context
    if response.status_code != 200:
        return f"LLM request failed: {response.status_code}", context
    resp_json = response.json()
    choices = resp_json.get("choices")
    if not choices or not isinstance(choices, list):
        return "No choices found in LLM response", context
    first_choice = choices[0]
    message = first_choice.get("message") or first_choice.get("text")
    if isinstance(message, dict):
        output = message.get("content") or message.get("text")
    else:
        output = message
    return output or "No response text found.", context


@app.route("/", methods=["GET", "POST"])
def chat():
    question = ""
    answer = ""
    context = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer, context = get_answer(question)
        else:
            answer = "Please ask a question."
    return render_template_string(HTML, question=question, answer=answer, context=context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

