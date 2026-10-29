
import streamlit as st

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

st.title("🧠 Intelligent RAG Model Laboratory")
st.caption(
    "Exploring how VAE, Naive Bayes and HMM can enhance "
    "technical document retrieval."
)

st.sidebar.title("MODEL LAB")
st.sidebar.caption("Technical AI Research Demonstrator")
st.sidebar.info("Dataset: Synthetic maintenance data")

overview, vae, nb, hmm, evaluation = st.tabs([
    "🏠 RAG Control Center",
    "📈 VAE Laboratory",
    "🧭 Naive Bayes",
    "🔄 HMM Laboratory",
    "📊 Evaluation",
])

with overview:
    st.subheader("Intelligent Retrieval Overview")
    st.write(
        "Explore how machine-learning models can provide "
        "additional context to a technical document retrieval system."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Models", "3")
    c2.metric("Document Corpus", "Synthetic")
    c3.metric("Retrieval Mode", "Baseline")

    st.info(
        "The first milestone establishes the interface. "
        "Next, we will add searchable documents and real model calculations."
    )

with vae:
    st.subheader("VAE Anomaly Laboratory")
    st.write("Coming next: synthetic telemetry and anomaly scoring.")
    st.slider("Temperature (°C)", 20, 100, 55)
    st.slider("Packet loss (%)", 0, 50, 5)
    st.slider("Restart count", 0, 10, 1)

with nb:
    st.subheader("Naive Bayes Query Intelligence")
    st.text_area(
        "Enter a technical maintenance question",
        placeholder="Why is the equipment repeatedly restarting?",
    )
    st.caption("Next: classification probabilities and query routing.")

with hmm:
    st.subheader("Hidden Markov Model State Tracker")
    st.write("Next: observation sequences and inferred equipment states.")

with evaluation:
    st.subheader("Comparative RAG Evaluation")
    st.write(
        "Next: compare baseline retrieval with model-assisted retrieval "
        "on a shared synthetic test set."
    )
