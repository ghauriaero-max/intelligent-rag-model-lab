
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from documents import DOCUMENTS

st.set_page_config(
    page_title="Intelligent RAG Model Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5edf8;
}
[data-testid="stSidebar"] {
    background: #111c2e;
}
.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}
div[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #263b56;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Document retrieval engine ----------

@st.cache_resource
def build_retriever():
    texts = [doc["title"] + " " + doc["text"] for doc in DOCUMENTS]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    document_vectors = vectorizer.fit_transform(texts)
    return vectorizer, document_vectors


def retrieve_documents(question, top_k=3):
    vectorizer, document_vectors = build_retriever()
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector, document_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        results.append({
            **DOCUMENTS[index],
            "score": float(scores[index]),
        })

    return results


# ---------- Application layout ----------

st.title("🧠 Intelligent RAG Model Laboratory")
st.caption(
    "A research demonstrator for ML-assisted technical document retrieval"
)

st.sidebar.title("MODEL LAB")
st.sidebar.caption("Technical AI Research Demonstrator")
st.sidebar.success("Synthetic corpus loaded")
st.sidebar.metric("Indexed documents", len(DOCUMENTS))

overview, vae, nb, hmm, evaluation = st.tabs([
    "🏠 RAG Control Center",
    "📈 VAE Laboratory",
    "🧭 Naive Bayes",
    "🔄 HMM Laboratory",
    "📊 Evaluation",
])

with overview:
    st.subheader("Technical Document Intelligence")
    st.write(
        "Ask a question about the synthetic maintenance corpus. "
        "The retrieval engine ranks passages by TF-IDF cosine similarity."
    )

    question = st.text_input(
        "Technical question",
        placeholder="Why might equipment restart unexpectedly?",
        key="rag_question",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.selectbox("Maximum results", [1, 2, 3, 5], index=2)
    with col2:
        st.caption("Search method: TF-IDF + cosine similarity")

    search_clicked = st.button(
        "🔎 Retrieve technical evidence",
        type="primary",
        use_container_width=True,
    )

    if search_clicked:
        if not question.strip():
            st.warning("Enter a technical question first.")
        else:
            results = retrieve_documents(question, top_k)

            if not results:
                st.warning(
                    "No matching terms were found in the corpus. "
                    "Try different wording."
                )
            else:
                st.subheader("Retrieved evidence")

                c1, c2 = st.columns(2)
                c1.metric("Matching passages", len(results))
                c2.metric(
                    "Best similarity score",
                    f"{results[0]['score']:.3f}",
                )

                st.caption(
                    "Scores are relative TF-IDF cosine similarities, "
                    "not probabilities or calibrated confidence."
                )

                for rank, result in enumerate(results, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**{rank}. {result['title']}**"
                        )
                        st.caption(
                            f"Source: {result['id']} · "
                            f"Category: {result['category']} · "
                            f"Similarity: {result['score']:.3f}"
                        )
                        st.write(result["text"])

                st.info(
                    "This version retrieves evidence only. "
                    "It does not yet generate an answer with an LLM."
                )

    st.divider()
    st.subheader("Retrieval corpus overview")

    df = pd.DataFrame([
        {
            "Document ID": d["id"],
            "Title": d["title"],
            "Category": d["category"],
            "Characters": len(d["text"]),
        }
        for d in DOCUMENTS
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)

with vae:
    st.subheader("VAE Anomaly Laboratory")
    st.write("Next milestone: train a small VAE on synthetic telemetry.")
    st.slider("Temperature (°C)", 20, 100, 55)
    st.slider("Packet loss (%)", 0, 50, 5)
    st.slider("Restart count", 0, 10, 1)

with nb:
    st.subheader("Naive Bayes Query Intelligence")
    st.text_area(
        "Enter a technical maintenance question",
        placeholder="Why is the equipment repeatedly restarting?",
    )
    st.caption("Next: intent classification and retrieval routing.")

with hmm:
    st.subheader("Hidden Markov Model State Tracker")
    st.write("Next: observation sequences and inferred equipment states.")

with evaluation:
    st.subheader("Comparative RAG Evaluation")
    st.write(
        "Next: evaluate baseline retrieval against model-assisted retrieval."
    )
