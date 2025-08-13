# Student Services Chatbot Project

An AI‑powered chatbot designed to assist students with navigating key services at Conestoga College — including financial aid, academic support, IT, and more.

---

## **Team & Context**

**Team 5** — Collaborative project developed as part of:  
- **PROG8245-25S-Sec1 — Machine Learning Programming**  
- **CSCN8010-25S-Sec1 — Foundations of Machine Learning Frameworks**  

**Team Members:**  
- **Mandeep Singh** — Data Preprocessing & Model Development  
- **Kumari Nikitha Singh** — NLP Pipeline & Intent Classification  
- **Krishna Reddy** — LLM Integration, Retrieval Logic & Fallback Strategies  

This project integrates **traditional NLP techniques** with **modern LLM-based methods** to deliver a robust, multi-layered Student Services Chatbot, implementing **retrieval ranking, intent detection, emotion analysis, and intelligent fallback strategies** for high accuracy and natural user interaction.


---

##  Project Overview

This chatbot offers an interactive, intelligent assistant that:

- Predicts **user intent** (e.g., financial, technical, contact),
- Detects **emotional tone** to handle distressed queries with empathy,
- Retrieves answers from a curated **FAQ database** using TF‑IDF,
- Falls back gracefully to an **LLM** when FAQ confidence is low,
- Escalates to an **advisor scheduler** if unresolved.

The UI is served via **Streamlit**, alongside a Jupyter notebook for data preprocessing, model training, and evaluation.

---

##  Architecture & Pipeline

```plaintext
User Query
    ↓
Emotion Detection → (if negative) empathetic response
    ↓
Intent Classification
    ↓
TF‑IDF FAQ Retrieval (cosine similarity)
    ↓
[ if score < threshold → LLM Fallback ]
    ↓
Display Answer + [Optional] Advisor Escalation
```

- **Emotion Detection**: Filters negative sentiment to prompt empathy.
- **Intent Classification**: Maps queries to categories via logistic regression.
- **FAQ Retrieval**: Matches FAQs using cosine similarity on TF‑IDF vectors.
- **LLM Fallback**: Uses Gemini/OpenRouter for unmatched or general queries.
- **Advisor Escalation**: Includes a link to scheduler for unresolved cases.

---

##  Repository Structure

```
Student-Services-Chatbot-Project/
├── app.py                   # Streamlit chatbot application
├── notebook/                # Jupyter notebook (data pipeline + training)
├── model/                   # Trained intent model & vectorizer (.pkl)
├── artifacts/               # TF‑IDF vectorizer & matrix (.pkl)
├── data/                    # Cleaned/faqs.csv, raw chunks, etc.
├── reports/                 # HTML technical report & evaluation
├── presentation.pptx        # Project presentation slides
├── requirements.txt         # Python dependencies
└── README.md                # Documentation (this file)
```

---

##  Getting Started

### Clone the repo
```bash
git clone https://github.com/kittuai/Student-Services-Chatbot-Project.git
cd Student-Services-Chatbot-Project
```

### Setup environment
```bash
conda create -n ssa python=3.10 -y
conda activate ssa
# or using pip:
pip install -r requirements.txt
```


---

##  How to Run

- **Streamlit app (chatbot interface):**
  ```bash
  streamlit run app.py
  ```
  Access at `http://localhost:8501`

- **Jupyter Notebook (training & evaluation):**
  ```bash
  jupyter notebook
  ```
  Then open the notebook in the `notebook/` folder.

---

##  Evaluation Metrics

We used:

- **Precision @5**: e.g., 0.92  
- **Mean Reciprocal Rank (MRR)**: e.g., 0.87  

Full details and scoring examples can be found in the HTML report under `reports/`.

---

##  Technical Highlights

| Component                | Key Details                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Scraping & Chunking**  | Extracts FAQs from Conestoga web pages, tokenizes and chunks text.          |
| **TF‑IDF Retrieval**      | 1–2 n‑gram vectorization for matching FAQ questions.                        |
| **Intent Classifier**     | Logistic regression with ~94% accuracy.                                     |
| **Emotion Detection**     | LLM-based labeler to detect emotional tone for empathy.                    |
| **LLM Fallback**          | Generates fallback answers when FAQ scores are weak or intent is general.  |
| **Streamlit UI**          | Clean chat interface with metadata (emotion, intent, source) + top‑5 answers.|

---

##  Documentation & Presentation

- **HTML Project Report** → `reports/project_report.html`  
- **Presentation Slides** → `presentation.pptx` (key diagrams and flow)  
- **Notebook** → `notebook/analysis.ipynb`

---

##  Future Enhancements

- Deploy to Conestoga portal or LMS.
- Support multilingual queries.
- Add voice interaction and accessibility features.
- Adapt fine-tuned LLMs for domain specificity.

---

##  Contact & License

**Project Repository**: [Student-Services-Chatbot-Project](https://github.com/kittuai/Student-Services-Chatbot-Project)  
**Authors**: Mandeep Singh, Nikitha Singh, Krishna Reddy  
**License**: MIT (see LICENSE file)

---

Thank you for exploring our Student Services Chatbot—built to bridge students with help faster, smarter, and with understanding.
