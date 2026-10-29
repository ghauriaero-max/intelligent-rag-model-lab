
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from documents import DOCUMENTS


# ============================================================
# 1. PAGE CONFIGURATION AND CUSTOM STYLING
# ============================================================

st.set_page_config(
    page_title="Intelligent RAG Model Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5edf8;
}
[data-testid="stSidebar"] {
    background-color: #111c2e;
}
.block-container {
    padding-top: 1.8rem;
    max-width: 1500px;
}
div[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #263b56;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #a9bdd5;
}
h1, h2, h3 {
    color: #e5edf8;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #263b56;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. SYNTHETIC NAIVE BAYES TRAINING DATA
# ============================================================

TRAINING_DATA = [
    ("What causes packet loss and poor communication?",
     "troubleshooting"),
    ("How can I investigate an intermittent connection?",
     "troubleshooting"),
    ("Why is the communication link unstable?",
     "troubleshooting"),
    ("What causes degraded link quality?",
     "troubleshooting"),
    ("Why does the equipment restart repeatedly?",
     "fault_isolation"),
    ("How can I investigate unexpected restarts?",
     "fault_isolation"),
    ("Could unstable power cause a system reset?",
     "fault_isolation"),
    ("What can cause repeated equipment resets?",
     "fault_isolation"),
    ("How should I inspect equipment regularly?",
     "preventive_maintenance"),
    ("What are the routine inspection steps?",
     "preventive_maintenance"),
    ("How should operating temperature be monitored?",
     "preventive_maintenance"),
    ("What should a preventive inspection include?",
     "preventive_maintenance"),
]


# ============================================================
# 3. BASELINE RAG RETRIEVAL ENGINE
# ============================================================

@st.cache_resource
def build_retriever():
    texts = [
        doc["title"] + " " + doc["text"]
        for doc in DOCUMENTS
    ]

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
        query_vector,
        document_vectors,
    ).flatten()

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        result = DOCUMENTS[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


# ============================================================
# 4. NAIVE BAYES QUERY CLASSIFIER
# ============================================================

@st.cache_resource
def build_classifier():
    questions = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            MultinomialNB(),
        ),
    ])

    model.fit(questions, labels)

    return model


def classify_query(question):
    model = build_classifier()

    predicted_class = model.predict([question])[0]
    probabilities = model.predict_proba([question])[0]

    probability_map = dict(
        zip(model.classes_, probabilities)
    )

    return predicted_class, probability_map


# ============================================================
# 5. CATEGORY-AWARE RETRIEVAL
# ============================================================

def retrieve_with_category(question, predicted_category, top_k=5):
    # Retrieve the full available document ranking first.
    results = retrieve_documents(
        question,
        top_k=len(DOCUMENTS),
    )

    # A transparent demonstration heuristic.
    category_bonus = 0.15

    for result in results:
        bonus = (
            category_bonus
            if result["category"] == predicted_category
            else 0.0
        )

        result["category_bonus"] = bonus
        result["adjusted_score"] = result["score"] + bonus

    results.sort(
        key=lambda item: item["adjusted_score"],
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# 6. SHARED DISPLAY HELPER
# ============================================================

def display_results(results, show_adjusted=False):
    if not results:
        st.warning(
            "No matching terms were found in the document corpus. "
            "Try rephrasing your question."
        )
        return

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(
                f"**{rank}. {result['title']}**"
            )

            metadata = (
                f"Source: {result['id']}  |  "
                f"Category: {result['category']}  |  "
                f"TF-IDF similarity: {result['score']:.3f}"
            )

            if show_adjusted:
                metadata += (
                    f"  |  Category bonus: "
                    f"{result['category_bonus']:.2f}"
                    f"  |  Adjusted score: "
                    f"{result['adjusted_score']:.3f}"
                )

            st.caption(metadata)
            st.write(result["text"])


# ============================================================
# 7. APPLICATION HEADER AND SIDEBAR
# ============================================================

st.title("🧠 Intelligent RAG Model Laboratory")

st.caption(
    "Investigating how machine-learning models can improve "
    "technical document retrieval."
)

st.sidebar.title("MODEL LAB")
st.sidebar.caption("Technical AI Research Demonstrator")

st.sidebar.success("Synthetic corpus loaded")
st.sidebar.metric("Indexed documents", len(DOCUMENTS))
st.sidebar.metric("ML models planned", 3)

st.sidebar.divider()

st.sidebar.markdown("### Research workflow")
st.sidebar.markdown(
    """
    1. Establish baseline retrieval
    2. Classify query intent
    3. Infer temporal equipment state
    4. Detect anomalous telemetry
    5. Compare retrieval performance
    """
)

st.sidebar.caption(
    "Prototype uses synthetic data. No external LLM is connected."
)


# ============================================================
# 8. FIVE MAIN APPLICATION TABS
# ============================================================

overview, vae, nb, hmm, evaluation = st.tabs([
    "🏠 RAG Control Center",
    "📈 VAE Laboratory",
    "🧭 Naive Bayes",
    "🔄 HMM Laboratory",
    "📊 Evaluation",
])


# ============================================================
# TAB 1: RAG CONTROL CENTER
# ============================================================

with overview:
    st.subheader("Technical Document Intelligence")

    st.write(
        "Enter a technical question to retrieve relevant passages "
        "from the synthetic maintenance document collection."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Document collection",
        len(DOCUMENTS),
    )

    col2.metric(
        "Retrieval method",
        "TF-IDF",
    )

    col3.metric(
        "Generation",
        "Not connected",
    )

    question = st.text_input(
        "Ask a technical question",
        placeholder=(
            "Why might equipment restart unexpectedly?"
        ),
        key="rag_question",
    )

    search_col, topk_col = st.columns([2, 1])

    with topk_col:
        top_k = st.selectbox(
            "Maximum results",
            [1, 2, 3, 5],
            index=2,
            key="rag_top_k",
        )

    with search_col:
        st.caption(
            "Baseline retrieval: TF-IDF vectors + cosine similarity"
        )

    if st.button(
        "🔎 Retrieve technical evidence",
        type="primary",
        use_container_width=True,
        key="rag_search",
    ):
        if not question.strip():
            st.warning("Please enter a technical question.")

        else:
            results = retrieve_documents(question, top_k)

            if not results:
                st.warning(
                    "No matching terms were found. "
                    "Try different wording."
                )

            else:
                best_score = results[0]["score"]

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Retrieved passages",
                    len(results),
                )

                metric2.metric(
                    "Best similarity score",
                    f"{best_score:.3f}",
                )

                st.caption(
                    "Similarity is not a probability or a calibrated "
                    "measure of answer correctness."
                )

                st.markdown("### Retrieved evidence")

                display_results(results)

                st.info(
                    "This is the baseline RAG retrieval stage. "
                    "It retrieves source passages but does not yet "
                    "generate an answer using a language model."
                )

    st.divider()

    st.subheader("Document Corpus Explorer")

    corpus_df = pd.DataFrame([
        {
            "Document ID": doc["id"],
            "Title": doc["title"],
            "Category": doc["category"],
            "Characters": len(doc["text"]),
        }
        for doc in DOCUMENTS
    ])

    st.dataframe(
        corpus_df,
        use_container_width=True,
        hide_index=True,
    )

    category_counts = (
        corpus_df["Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Document count")
    )

    st.markdown("### Corpus composition")

    st.bar_chart(
        category_counts.set_index("Category")
    )


# ============================================================
# TAB 2: VAE LABORATORY - INITIAL TELEMETRY INTERFACE
# ============================================================

with vae:
    st.subheader("VAE Anomaly Laboratory")

    st.write(
        "Explore synthetic equipment telemetry. "
        "These controls are a UI prototype; the VAE model "
        "and trained anomaly detector are not implemented yet."
    )

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Telemetry controls")

        temperature = st.slider(
            "Temperature (°C)",
            min_value=20,
            max_value=100,
            value=55,
            key="vae_temperature",
        )

        packet_loss = st.slider(
            "Packet loss (%)",
            min_value=0,
            max_value=50,
            value=5,
            key="vae_packet_loss",
        )

        restarts = st.slider(
            "Restart count",
            min_value=0,
            max_value=10,
            value=1,
            key="vae_restarts",
        )

        st.caption(
            "Synthetic values for demonstration only."
        )

    with right:
        st.markdown("### Current telemetry")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Temperature",
            f"{temperature} °C",
        )

        m2.metric(
            "Packet loss",
            f"{packet_loss}%",
        )

        m3.metric(
            "Restarts",
            restarts,
        )

        # Illustrative rule-based indicator, not a VAE score.
        illustrative_flag = (
            temperature >= 80
            or packet_loss >= 20
            or restarts >= 4
        )

        if illustrative_flag:
            st.warning(
                "Illustrative threshold flag: one or more telemetry "
                "values exceed the demo thresholds."
            )
        else:
            st.info(
                "Illustrative threshold flag: no demo threshold exceeded."
            )

        st.caption(
            "This is a simple rule-based indicator, NOT a VAE "
            "reconstruction error or a learned anomaly score."
        )

    st.markdown("### Potential RAG integration")

    st.write(
        "A future trained VAE could flag unusual telemetry. "
        "Its output could then prioritize searches for relevant "
        "maintenance procedures. This relationship must be evaluated "
        "against baseline retrieval."
    )


# ============================================================
# TAB 3: NAIVE BAYES QUERY INTELLIGENCE
# ============================================================

with nb:
    st.subheader("Naive Bayes Query Intelligence")

    st.write(
        "Predict the intent of a technical question, inspect class "
        "probabilities, and compare baseline retrieval with "
        "category-aware reranking."
    )

    nb_question = st.text_input(
        "Enter a technical maintenance question",
        placeholder=(
            "Why does the equipment repeatedly restart?"
        ),
        key="nb_question",
    )

    if st.button(
        "🧭 Classify and retrieve",
        type="primary",
        use_container_width=True,
        key="nb_run",
    ):
        if not nb_question.strip():
            st.warning("Please enter a question.")

        else:
            predicted, probabilities = classify_query(
                nb_question
            )

            st.markdown("### Predicted query category")

            st.success(
                predicted.replace("_", " ").title()
            )

            prob_df = pd.DataFrame({
                "Category": list(probabilities.keys()),
                "Probability": list(probabilities.values()),
            }).sort_values(
                "Probability",
                ascending=False,
            )

            st.markdown("### Class probabilities")

            st.bar_chart(
                prob_df.set_index("Category")
            )

            st.caption(
                "Probabilities come from a small synthetic training set. "
                "They are not calibrated confidence estimates."
            )

            baseline_results = retrieve_documents(
                nb_question,
                top_k=len(DOCUMENTS),
            )

            reranked_results = retrieve_with_category(
                nb_question,
                predicted,
                top_k=5,
            )

            st.markdown("### Retrieval comparison")

            baseline_col, routed_col = st.columns(2)

            with baseline_col:
                st.markdown("#### Baseline ranking")
                st.caption(
                    "Ranked using text similarity alone."
                )

                display_results(
                    baseline_results[:3]
                )

            with routed_col:
                st.markdown("#### Category-aware ranking")
                st.caption(
                    "Text similarity plus a fixed category bonus."
                )

                display_results(
                    reranked_results[:3],
                    show_adjusted=True,
                )

            st.info(
                "The category bonus is a tunable demonstration heuristic. "
                "It is not a learned parameter, and it may worsen ranking "
                "if the predicted category is wrong."
            )

            with st.expander("Inspect training examples"):
                training_df = pd.DataFrame(
                    TRAINING_DATA,
                    columns=["Training question", "Label"],
                )

                st.dataframe(
                    training_df,
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# TAB 4: HMM LABORATORY - INITIAL INTERFACE
# ============================================================

with hmm:
    st.subheader("Hidden Markov Model State Tracker")

    st.write(
        "Explore a possible temporal-state model for equipment "
        "observations. The current version is an interface prototype; "
        "a trained HMM and Viterbi decoder are not implemented yet."
    )

    observation_options = [
        "Normal temperature",
        "Elevated temperature",
        "High packet loss",
        "Repeated restart",
    ]

    observations = st.multiselect(
        "Select an observation sequence",
        observation_options,
        default=[
            "Normal temperature",
            "Elevated temperature",
            "High packet loss",
        ],
        key="hmm_observations",
    )

    st.markdown("### Illustrative state progression")

    if observations:
        state_labels = []

        for observation in observations:
            if observation == "Normal temperature":
                state_labels.append("Possible normal state")
            elif observation == "Elevated temperature":
                state_labels.append("Possible degraded state")
            elif observation == "High packet loss":
                state_labels.append("Possible communication degradation")
            else:
                state_labels.append("Possible fault state")

        for index, (obs, state) in enumerate(
            zip(observations, state_labels),
            start=1,
        ):
            st.markdown(
                f"**Step {index}:** {obs}  →  *{state}*"
            )

        st.caption(
            "These are illustrative mappings, not HMM-inferred states. "
            "An actual HMM requires defined transition/emission "
            "probabilities and a decoding algorithm."
        )

    else:
        st.info(
            "Select observations to view the illustrative sequence."
        )

    st.markdown("### Potential RAG integration")

    st.write(
        "An actual HMM could infer a hidden state from a sequence "
        "of observations. That state could become additional retrieval "
        "context, alongside the user's question."
    )


# ============================================================
# TAB 5: COMPARATIVE EVALUATION
# ============================================================

with evaluation:
    st.subheader("Comparative RAG Evaluation")

    st.write(
        "Evaluate baseline retrieval against model-assisted retrieval "
        "using a fixed set of test questions and known relevant documents."
    )

    st.markdown("### Current prototype status")

    status_df = pd.DataFrame([
        {
            "Component": "TF-IDF retrieval",
            "Status": "Implemented",
            "Role": "Baseline passage retrieval",
        },
        {
            "Component": "Naive Bayes",
            "Status": "Implemented",
            "Role": "Query intent classification",
        },
        {
            "Component": "Category-aware reranking",
            "Status": "Implemented",
            "Role": "Heuristic category bonus",
        },
        {
            "Component": "VAE",
            "Status": "Not implemented",
            "Role": "Future anomaly detection",
        },
        {
            "Component": "HMM",
            "Status": "Not implemented",
            "Role": "Future temporal-state inference",
        },
        {
            "Component": "Generative LLM",
            "Status": "Not connected",
            "Role": "Future answer generation",
        },
    ])

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Retrieval evaluation metrics")

    st.write(
        """
        - **Recall@k:** Does the top-k result set include relevant documents?
        - **MRR:** How highly ranked is the first relevant document?
        - **Precision@k:** What fraction of the top-k documents are relevant?
        """
    )

    st.warning(
        "Do not interpret a higher similarity score as proof of better "
        "retrieval. A proper evaluation needs labeled test queries, "
        "relevant-document judgments, and a comparison on the same cases."
    )

    st.markdown("### Suggested next experiment")

    st.write(
        "Create a held-out set of synthetic maintenance questions, "
        "label the relevant document IDs, and compare baseline TF-IDF "
        "against category-aware reranking using the same evaluation set."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent RAG Model Laboratory | Research prototype | "
    "Synthetic documents only | Not for operational maintenance decisions"
)
