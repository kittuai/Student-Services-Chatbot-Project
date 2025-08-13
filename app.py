import os
import re
import pickle
import requests
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG (hard-coded key)
# ========================
API_KEY = "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"
MODEL = "meta-llama/llama-3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
SCHEDULING_LINK = "https://studentsuccess.conestogac.on.ca/meet"

SIMILARITY_THRESHOLD = 0.30
TOP_N_FAQ = 5
SYSTEM_INSTRUCTIONS = (
    "You are an SSA chatbot for a Canadian college. "
    "Be precise and concise. Answer in 1–3 short sentences. "
    "Do not use the word 'Conestoga' in your answers."
)

# ========================
# LOAD ASSETS
# ========================
base = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_assets():
    with open(os.path.join(base, "model", "intent_classifier.pkl"), "rb") as f:
        intent_model = pickle.load(f)
    with open(os.path.join(base, "model", "intent_vectorizer.pkl"), "rb") as f:
        intent_vectorizer = pickle.load(f)
    with open(os.path.join(base, "artifacts", "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(os.path.join(base, "artifacts", "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    df = pd.read_csv(os.path.join(base, "data", "faqs_relabelled.csv"))
    return intent_model, intent_vectorizer, tfidf_vectorizer, tfidf_matrix, df

try:
    intent_model, intent_vectorizer, tfidf_vectorizer, tfidf_matrix, df = load_assets()
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

# ========================
# HELPERS (LLM + utils)
# ========================
def _post_openrouter(messages, temperature=0.1, max_tokens=220):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Service unavailable. Try again later."

def clean_branding(text: str) -> str:
    """Remove/replace the word 'conestoga' everywhere."""
    return re.sub(r"\bconestoga\b", "the college", text, flags=re.IGNORECASE)

def detect_intent(text: str) -> str:
    v = intent_vectorizer.transform([text])
    return intent_model.predict(v)[0]

def match_faq_tfidf(text: str, top_n: int = TOP_N_FAQ):
    """Rank FAQs via TF-IDF, skip any row that mentions 'conestoga'."""
    qv = tfidf_vectorizer.transform([text])
    scores = cosine_similarity(qv, tfidf_matrix).flatten()
    idx = scores.argsort()[::-1]

    out = []
    for i in idx:
        q_text = df.iloc[i]["question"]
        a_text = df.iloc[i]["answer"]
        score = float(scores[i])
        if re.search(r"\bconestoga\b", q_text, re.IGNORECASE) or re.search(r"\bconestoga\b", a_text, re.IGNORECASE):
            continue
        out.append({"q": clean_branding(q_text), "a": clean_branding(a_text), "score": score})
        if len(out) >= top_n:
            break
    return out

def detect_emotion(text: str) -> str:
    """Return: positive | negative | neutral."""
    prompt = (
        "Detect emotion. Reply with one word: positive, negative, or neutral.\n"
        f"Text: {text}"
    )
    label = _post_openrouter(
        [
            {"role": "system", "content": "Return only one word: positive, negative, or neutral."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0, max_tokens=3
    ).lower()
    if "positive" in label: return "positive"
    if "negative" in label: return "negative"
    return "neutral"

def call_llm(query: str) -> str:
    raw = _post_openrouter(
        [{"role": "system", "content": SYSTEM_INSTRUCTIONS},
         {"role": "user", "content": query}],
        temperature=0.1, max_tokens=220
    )
    return clean_branding(raw)

def _recent_empathy_banlist(n: int = 3):
    """Collect last n empathy messages to reduce repetition."""
    if "messages" not in st.session_state:
        return []
    bans = []
    for m in reversed(st.session_state.messages):
        if m.get("role") == "assistant" and m.get("meta", {}).get("source") == "llm-empathy":
            bans.append(m.get("content", ""))
            if len(bans) >= n:
                break
    return bans

def call_llm_empathy(user_text: str, intent_hint: str = "general") -> str:
    """
    Produce a short, empathetic response (1–3 sentences) using the LLM.
    Varies wording by banning recent empathy phrases; lightly tailors to detected intent.
    """
    banned = _recent_empathy_banlist(3)
    banned_join = " | ".join(banned) if banned else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a supportive student services assistant. "
                "The user sounds distressed or negative. "
                "Respond with empathy first, in 1–3 short sentences. "
                "Acknowledge feelings, offer one concrete next step, and (optionally) suggest meeting an advisor: "
                f"{SCHEDULING_LINK}. "
                "Vary your wording—do not repeat prior phrasing from the list below. "
                "Be precise and concise. Do not mention the word 'Conestoga'. "
                "Do not provide detailed policy info—focus on support.\n\n"
                f"Do not use these phrases verbatim: {banned_join}"
            ).strip(),
        },
        {
            "role": "user",
            "content": (
                f"The student wrote: {user_text}\n"
                f"Detected intent: {intent_hint}\n"
                "Write a short empathetic response that acknowledges the feeling and suggests one next step."
            ),
        },
    ]
    raw = _post_openrouter(messages, temperature=0.3, max_tokens=120)
    return clean_branding(raw)

def is_college_related_llm(text: str) -> bool:
    """
    Use LLM to classify if the query is college-related.
    Returns True only if the model replies exactly 'college'.
    """
    prompt = (
        "Classify the user's query. Reply with exactly one word:\n"
        "- 'college' if it is about higher education topics such as fees, tuition, courses, programs, registration, "
        "admissions, campus, faculty, advising, student services, exams, library, deadlines, housing, scholarships, loans, "
        "ID cards, portals, or accounts.\n"
        "- 'other' for anything else.\n\n"
        f"Query: {text}"
    )
    try:
        label = _post_openrouter(
            [
                {"role": "system", "content": "Return only one word: college or other."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0, max_tokens=3
        ).strip().lower()
        return label == "college"
    except Exception:
        return False  # safe default

# ========================
# OFF-RAMP (centralized)
# ========================
def decide_response(user_text):
    """
    Decide route: 'tf-idf' | 'llm' | 'llm-empathy'.
    - If emotion is negative → empathetic LLM reply (we'll ALSO add an LLM answer in the UI layer).
    - Else if no strong TF-IDF match and intent is general → LLM.
    - Else if no matches → LLM.
    - Else TF-IDF.
    """
    emotion = detect_emotion(user_text)
    intent = detect_intent(user_text)
    faqs = match_faq_tfidf(user_text, TOP_N_FAQ)
    best_score = faqs[0]["score"] if faqs else 0.0
    has_strong_match = best_score >= SIMILARITY_THRESHOLD

    decision = {
        "path": None,
        "answer": "",
        "faqs": faqs,          # keep faqs even for negative, so we can show Top-5 on the second message
        "emotion": emotion,
        "intent": intent
    }

    # Negative emotion → empathetic LLM (UI will add the second LLM answer + table)
    if emotion == "negative":
        decision["path"] = "llm-empathy"
        decision["answer"] = call_llm_empathy(user_text, intent_hint=intent)
        return decision

    # LLM if weak match and general intent
    if not has_strong_match and intent.lower() == "general":
        decision["path"] = "llm"
        decision["answer"] = call_llm(user_text)
        return decision

    # LLM fallback if no matches
    if not faqs:
        decision["path"] = "llm"
        decision["answer"] = call_llm(user_text)
        return decision

    # Default: TF-IDF best answer
    decision["path"] = "tf-idf"
    decision["answer"] = faqs[0]["a"]
    return decision

# ========================
# UI (minimal, fixed title position)
# ========================
st.set_page_config(page_title="SSA Chat", layout="centered")

# Add CSS to push content down and center title
st.markdown("""
<style>
.block-container {
    max-width: 760px;
    padding-top: 40px;  /* Pushes content down so title is visible */
}
.stChatMessage { padding: 4px 0; }
.stChatMessage .stMarkdown { font-size: 0.92rem; line-height: 1.35; }
.meta { color:#6b7280; font-size:0.80rem; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# Centered title and subtitle
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>Student Service Assistant</h1>
    <p style='text-align: center; color: gray; font-size: 0.95rem;'>
    Ask about student services, policies, or processes.
    </p>
    """,
    unsafe_allow_html=True
)

# Initialize history once with greeting
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. Ask about student services, policies, or processes.",
            "meta": {},
            "faqs": [],
            "query": "",
            "college_related": False
        }
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta", {})
        if m["role"] == "assistant" and meta:
            # Show only emotion, intent, source
            st.markdown(
                f"<div class='meta'>emotion: {meta.get('emotion','-')} · intent: {meta.get('intent','-')} · source: {meta.get('source','-')}</div>",
                unsafe_allow_html=True
            )
        # Show Top-5 only if LLM says user's query is college-related
        if m.get("faqs") and m.get("college_related", False):
            st.markdown("**Top 5 similar questions:**")
            df_view = pd.DataFrame(m["faqs"][:TOP_N_FAQ]).rename(columns={"q": "Question", "score": "Score"})
            df_view["Score"] = df_view["Score"].map(lambda x: f"{x:.2f}")
            st.dataframe(df_view[["Question", "Score"]], use_container_width=True, hide_index=True)

# Chat input
user_text = st.chat_input("Type your question")
if user_text:
    # Central decision
    decision = decide_response(user_text)

    # LLM classification for showing Top-5 table (college-related vs other)
    college_related = is_college_related_llm(user_text)

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "meta": {},
        "faqs": [],
        "query": user_text,
        "college_related": False
    })

    # If negative: 1) empathy (llm-empathy), 2) LLM answer with table
    if decision["emotion"] == "negative":
        # 1) Empathy message (meta shown here)
        st.session_state.messages.append({
            "role": "assistant",
            "content": decision["answer"],
            "meta": {
                "emotion": decision["emotion"],
                "intent": decision["intent"],
                "source": "llm-empathy"
            },
            "faqs": [],
            "query": user_text,
            "college_related": False
        })

        # 2) Actual concise LLM answer (no meta → cleaner look, table shown here)
        llm_answer = call_llm(user_text)
        faqs_view = [{"q": f["q"], "score": f["score"]} for f in decision["faqs"]]
        st.session_state.messages.append({
            "role": "assistant",
            "content": llm_answer,
            "meta": {},  # empty to avoid duplicate meta line
            "faqs": faqs_view,
            "query": user_text,
            "college_related": college_related
        })
        st.rerun()

    # Non-negative (original single response behavior)
    meta = {
        "emotion": decision["emotion"],
        "intent": decision["intent"],
        "source": decision["path"]
    }
    faqs_view = [{"q": f["q"], "score": f["score"]} for f in decision["faqs"]]

    st.session_state.messages.append({
        "role": "assistant",
        "content": decision["answer"],
        "meta": meta,
        "faqs": faqs_view,
        "query": user_text,
        "college_related": college_related
    })

    st.rerun()
    