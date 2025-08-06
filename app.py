import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIG ==========
API_KEY = "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"
MODEL = "meta-llama/llama-3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
CSV_PATH = r"C:\Users\kittu\Desktop\ssa\data\faqs.csv"

# ========== LOAD DATA ==========
faq_df = pd.read_csv(CSV_PATH).fillna("")
questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(questions)

# ========== HELPER FUNCTIONS ==========
def detect_emotion(text):
    neg = ["stressed", "confused", "lost", "worried", "sad", "help", "depressed"]
    pos = ["thank", "great", "awesome", "perfect", "good"]
    text = text.lower()
    if any(w in text for w in neg): return "negative"
    if any(w in text for w in pos): return "positive"
    return "neutral"

def is_greeting_only(text):
    greetings = ["hi", "hello", "hey", "howdy", "yo", "hola", "hai", "heyy", "sup"]
    return text.strip().lower() in greetings

def ask_llama(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    try:
        res = requests.post(API_URL, headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content'].strip()
    except:
        return None

def is_low_quality(text):
    if not text: return True
    bad = ["i don't know", "not sure", "no info", "unsure", "not found", "cannot help"]
    return any(b in text.lower() for b in bad)

def get_tfidf_answer(query, threshold=0.65):
    vague = ["help", "lost", "confused", "i am", "support", "need", "assist"]
    if any(p in query.lower() for p in vague) and len(query.split()) < 5:
        return None, None, 0.0, None
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = scores.argmax()
    if scores[best_idx] >= threshold:
        return answers[best_idx], questions[best_idx], round(scores[best_idx], 3), scores
    return None, None, round(scores[best_idx], 3), scores

def get_rag_answer(query):
    context = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions[:5], answers[:5]))
    prompt = f"""You are an SSA assistant at Conestoga College.

Context:
{context}

Now answer this question as a helpful assistant:
{query}"""
    return ask_llama(prompt)

def get_direct_answer(query, emotion):
    tone = "Use an empathetic tone." if emotion == "negative" else "Use a friendly tone."
    prompt = f"""{tone}

Please help this student:

{query}

If the topic is not related to Conestoga SSA, provide basic help and gently suggest asking about Student Services."""
    return ask_llama(prompt)

def generate_followup(query, answer):
    prompt = f"""User asked:
{query}

Assistant replied:
{answer}

Suggest a helpful, friendly follow-up like:
- Was this clear?
- Anything else I can help with?
Keep it short, natural, and do NOT repeat the answer."""
    reply = ask_llama(prompt)
    return reply.strip() if reply else "🙂 Do you have any other questions I can help with?"

def escalation():
    return ("It looks like you might need personal support.\n\n"
            "📅 [Book an SSA Advisor](https://studentsuccess.conestogac.on.ca/academic-guidance)\n"
            "📞 Or call: 555-123-4567")

# ========== STREAMLIT UI ==========
st.set_page_config("SSA Chatbot", layout="centered")
st.title("🎓 SSA Chatbot – Student Support Assistant")
st.caption("Ask about tuition, OSAP, advising, housing, or student services.")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Type your question here...")

# ========== GREETING HANDLING ==========
if user_query and is_greeting_only(user_query):
    prompt = """The user just greeted with 'hi' or 'hello'.

Write a short, friendly introduction message as a helpful SSA chatbot. Let them know they can ask about student services, OSAP, tuition, housing, etc. Avoid any specific assumption like scholarships unless mentioned."""
    greeting_reply = ask_llama(prompt)
    st.session_state.history.append({
        "query": user_query,
        "answer": greeting_reply,
        "method": "Greeting",
        "score": 0.0,
        "emotion": "neutral",
        "sims": None
    })
    user_query = ""  # Stop further processing

# ========== MAIN RESPONSE FLOW ==========
if user_query:
    emotion = detect_emotion(user_query)
    method = "None"
    final_reply = ""
    sims = None

    tfidf_ans, matched_q, score, sims = get_tfidf_answer(user_query)
    if tfidf_ans:
        final_reply = tfidf_ans
        method = "TF-IDF"
    else:
        rag_ans = get_rag_answer(user_query)
        if not is_low_quality(rag_ans):
            final_reply = rag_ans
            method = "RAG"
        else:
            direct_ans = get_direct_answer(user_query, emotion)
            if not is_low_quality(direct_ans):
                if "conestoga" not in direct_ans.lower():
                    direct_ans += "\n\n⚠ This may not be directly related to SSA. Want to ask something else?"
                final_reply = direct_ans
                method = "Direct"
            else:
                final_reply = escalation()
                method = "Escalation"

    followup = generate_followup(user_query, final_reply)
    combined = f"{final_reply}\n\n{followup}"

    st.session_state.history.append({
        "query": user_query,
        "answer": combined,
        "method": method,
        "score": score,
        "emotion": emotion,
        "sims": sims
    })

# ========== SHOW LATEST CHAT ONLY ==========
if st.session_state.history:
    latest = st.session_state.history[-1]
    st.markdown(f"🧑 **You:** {latest['query']}")
    st.markdown(f"🤖 **SSA Bot:** {latest['answer']}")
    st.caption(f"Source: {latest['method']} | Emotion: {latest['emotion']}")

# ========== DEBUG INFO ==========
if st.toggle("🛠 Show Debug Info", value=False):
    last = st.session_state.history[-1]
    if last["sims"] is not None:
        sims = last["sims"]
        top_idxs = sims.argsort()[::-1][:10]
        st.subheader("Top 10 TF-IDF Matches")
        debug_df = pd.DataFrame({
            "Rank": range(1, 11),
            "Question": [questions[i] for i in top_idxs],
            "Score": [round(sims[i], 3) for i in top_idxs]
        })
        st.dataframe(debug_df, use_container_width=True)

        matched = last["answer"]
        top_qs = [questions[i] for i in top_idxs]
        relevance = np.array([1 if q.strip() in matched else 0 for q in top_qs])

        def precision_at_k(r, k): return r[:k].sum() / k
        def map_score(r): return np.mean([precision_at_k(r, i+1) for i in range(len(r)) if r[i]]) if r.any() else 0.0
        def mrr_score(r): return 1.0 / (np.argmax(r) + 1) if r.any() else 0.0

        st.subheader("Ranking Metrics")
        st.write(f"📌 Precision@5: {precision_at_k(relevance, 5):.2f}")
        st.write(f"📌 MAP: {map_score(relevance):.2f}")
        st.write(f"📌 MRR: {mrr_score(relevance):.2f}")

# ========== CONVERSATION HISTORY ==========
if st.toggle("📜 Show Conversation History", value=False):
    for msg in st.session_state.history[:-1]:
        st.markdown(f"🧑 *You:* {msg['query']}")
        st.markdown(f"🤖 *Bot:* {msg['answer']}")
