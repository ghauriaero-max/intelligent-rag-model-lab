
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent RAG Model Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. HIGH-CONTRAST, READABLE DARK THEME
# ============================================================

st.markdown("""
<style>
/* Main canvas */
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
    font-size: 18px;
}

/* Main content width and spacing */
.block-container {
    max-width: 1600px;
    padding: 2rem 2.5rem 3rem 2.5rem;
}

/* Global text readability */
html, body, [class*="st-"] {
    font-size: 18px;
}

p, li, label, .stMarkdown, .stCaption {
    color: #E2E8F0;
    line-height: 1.65;
}

/* Headings */
h1 {
    color: #F8FAFC !important;
    font-size: 2.45rem !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
}

h2 {
    color: #F8FAFC !important;
    font-size: 1.85rem !important;
    font-weight: 750 !important;
    margin-top: 1.1rem !important;
}

h3 {
    color: #7DD3FC !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101C30;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #E2E8F0 !important;
    font-size: 17px !important;
}

/* Tabs: larger labels, clear contrast */
button[data-baseweb="tab"] {
    color: #DCE7F5 !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 15px 12px !important;
    min-height: 58px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #67E8F9 !important;
    border-bottom: 3px solid #22D3EE !important;
}

div[data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #475569;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #16263D;
    border: 1px solid #3B526F;
    border-radius: 14px;
    padding: 20px;
}

div[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
    font-size: 17px !important;
}

div[data-testid="stMetricValue"] {
    color: #67E8F9 !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #17263B !important;
    color: #FFFFFF !important;
    border: 1px solid #64748B !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    min-height: 48px;
}

div[data-baseweb="select"] > div {
    background-color: #17263B !important;
    color: #FFFFFF !important;
    border-color: #64748B !important;
    font-size: 17px !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #0E7490;
    color: #FFFFFF !important;
    border: 1px solid #67E8F9;
    border-radius: 9px;
    font-size: 17px !important;
    font-weight: 750 !important;
    min-height: 48px;
    padding: 8px 18px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #155E75;
    border-color: #A5F3FC;
}

/* Cards and containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111F33;
    border: 1px solid #3B526F;
    border-radius: 12px;
    padding: 10px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 17px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #475569;
    border-radius: 8px;
}

/* Captions remain readable */
.stCaption,
div[data-testid="stCaptionContainer"] {
    color: #B8C7DA !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}

/* Slider labels */
div[data-testid="stSlider"] label {
    color: #F1F5F9 !important;
    font-size: 17px !important;
}

/* Dividers */
hr {
    border-color: #334155;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Footer */
.lab-footer {
    color: #B8C7DA;
    background: #111F33;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-size: 15px;
    border: 1px solid #334155;
}

/* Responsive spacing on narrower screens */
@media (max-width: 900px) {
    .block-container {
        padding: 1rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }

    button[data-baseweb="tab"] {
        font-size: 14px !important;
        padding: 10px 6px !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD SYNTHETIC DOCUMENT CORPUS
# ============================================================

try:
    from documents import DOCUMENTS
except ImportError:
    st.error(
        "Cannot find documents.py. "
        "Make sure it exists in the same GitHub repository as app.py."
    )
    st.stop()


# ============================================================
# 4. NAIVE BAYES TRAINING DATA
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
# 5. BASELINE TF-IDF RETRIEVAL
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

    vectors = vectorizer.fit_transform(texts)

    return vectorizer, vectors


def retrieve_documents(question, top_k=3):
    vectorizer, vectors = build_retriever()

    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector, vectors
    ).flatten()

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        result = DOCUMENTS[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 6. NAIVE BAYES CLASSIFIER
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
        ("classifier", MultinomialNB()),
    ])

    model.fit(questions, labels)

    return model


def classify_query(question):
    model = build_classifier()

    predicted = model.predict([question])[0]
    probabilities = model.predict_proba([question])[0]

    probability_map = dict(
        zip(model.classes_, probabilities)
    )

    return predicted, probability_map


# ============================================================
# 7. CATEGORY-AWARE RERANKING
# ============================================================

def retrieve_with_category(question, category, top_k=5):
    results = retrieve_documents(
        question,
        top_k=len(DOCUMENTS),
    )

    for result in results:
        bonus = (
            0.15
            if result["category"] == category
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
# 8. DISPLAY RETRIEVAL RESULTS
# ============================================================

def display_results(results, show_adjusted=False):
    if not results:
        st.warning(
            "No matching passages found. Try rephrasing your question."
        )
        return

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(
                f"### {rank}. {result['title']}"
            )

            st.markdown(
                f"**Source:** `{result['id']}`  \n"
                f"**Category:** `{result['category']}`  \n"
                f"**TF-IDF similarity:** `{result['score']:.3f}`"
            )

            if show_adjusted:
                st.markdown(
                    f"**Category bonus:** "
                    f"`{result['category_bonus']:.2f}`  \n"
                    f"**Adjusted score:** "
                    f"`{result['adjusted_score']:.3f}`"
                )

            st.write(result["text"])


# ============================================================
# 9. HIDDEN MARKOV MODEL
# ============================================================

HMM_STATES = ["Normal", "Degraded", "Fault"]

HMM_OBSERVATIONS = [
    "Normal temperature",
    "Elevated temperature",
    "High packet loss",
    "Repeated restart",
]

HMM_START = np.array([0.80, 0.15, 0.05])

HMM_TRANSITION = np.array([
    [0.80, 0.18, 0.02],
    [0.15, 0.65, 0.20],
    [0.05, 0.20, 0.75],
])

HMM_EMISSION = np.array([
    [0.70, 0.10, 0.15, 0.05],
    [0.15, 0.45, 0.30, 0.10],
    [0.05, 0.15, 0.25, 0.55],
])


def viterbi_decode(observations):
    if not observations:
        return [], 0.0

    observation_indices = [
        HMM_OBSERVATIONS.index(obs)
        for obs in observations
    ]

    num_states = len(HMM_STATES)
    num_steps = len(observation_indices)

    log_start = np.log(HMM_START)
    log_transition = np.log(HMM_TRANSITION)
    log_emission = np.log(HMM_EMISSION)

    dp = np.full(
        (num_steps, num_states),
        -np.inf,
    )

    backpointer = np.zeros(
        (num_steps, num_states),
        dtype=int,
    )

    dp[0] = (
        log_start
        + log_emission[:, observation_indices[0]]
    )

    for t in range(1, num_steps):
        observation_index = observation_indices[t]

        for current_state in range(num_states):
            candidates = (
                dp[t - 1]
                + log_transition[:, current_state]
            )

            best_previous = int(np.argmax(candidates))

            dp[t, current_state] = (
                candidates[best_previous]
                + log_emission[
                    current_state, observation_index
                ]
            )

            backpointer[t, current_state] = best_previous

    last_state = int(np.argmax(dp[-1]))
    best_log_probability = float(dp[-1, last_state])

    path = [last_state]

    for t in range(num_steps - 1, 0, -1):
        path.append(
            int(backpointer[t, path[-1]])
        )

    path.reverse()

    decoded_states = [
        HMM_STATES[index]
        for index in path
    ]

    return decoded_states, best_log_probability


# ============================================================
# 10. APP HEADER
# ============================================================

st.title("🧠 Intelligent RAG Model Laboratory")

st.markdown(
    """
    Explore how machine-learning models can provide additional
    context for technical-document retrieval.
    """
)

st.caption(
    "Research prototype | Synthetic technical documents | "
    "High-contrast engineering dashboard"
)

st.sidebar.title("MODEL LAB")
st.sidebar.markdown("### Research Dashboard")
st.sidebar.divider()

st.sidebar.metric("Technical documents", len(DOCUMENTS))
st.sidebar.metric("Query categories", 3)
st.sidebar.metric("Implemented algorithms", 3)

st.sidebar.markdown("### System status")
st.sidebar.success("TF-IDF retrieval: Active")
st.sidebar.success("Naive Bayes: Active")
st.sidebar.success("HMM / Viterbi: Active")
st.sidebar.warning("VAE: Not yet implemented")
st.sidebar.info("LLM generation: Not connected")

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Research workflow**

    1. Retrieve document evidence
    2. Classify query intent
    3. Infer hidden equipment states
    4. Explore anomaly detection
    5. Evaluate retrieval quality
    """
)


# ============================================================
# 11. MAIN TABS
# ============================================================

overview, vae, nb, hmm, evaluation = st.tabs([
    "RAG CONTROL CENTER",
    "VAE LABORATORY",
    "NAIVE BAYES",
    "HMM STATE TRACKER",
    "EVALUATION",
])


# ============================================================
# TAB 1: RAG CONTROL CENTER
# ============================================================

with overview:
    st.header("RAG Control Center")

    st.write(
        "Ask a technical question and inspect the evidence retrieved "
        "from the synthetic maintenance document collection."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Indexed documents", len(DOCUMENTS))
    c2.metric("Retrieval engine", "TF-IDF")
    c3.metric("Answer generation", "Not connected")

    question = st.text_input(
        "Your technical question",
        placeholder="Why might equipment restart unexpectedly?",
        key="rag_question",
    )

    control_col, info_col = st.columns([1, 2])

    with control_col:
        top_k = st.selectbox(
            "Number of results",
            [1, 2, 3, 5],
            index=2,
            key="rag_top_k",
        )

    with info_col:
        st.markdown(
            "**Search method**  \n"
            "TF-IDF vectorization + cosine similarity"
        )

    if st.button(
        "SEARCH TECHNICAL DOCUMENTS",
        type="primary",
        use_container_width=True,
        key="rag_search",
    ):
        if not question.strip():
            st.warning("Enter a technical question first.")
        else:
            results = retrieve_documents(question, top_k)

            if results:
                metric_col1, metric_col2 = st.columns(2)

                metric_col1.metric(
                    "Matching passages",
                    len(results),
                )

                metric_col2.metric(
                    "Top similarity score",
                    f"{results[0]['score']:.3f}",
                )

                st.caption(
                    "Similarity is not a probability or a measure "
                    "of factual correctness."
                )

                st.header("Retrieved evidence")
                display_results(results)

                st.info(
                    "The current system retrieves evidence. It does not "
                    "yet generate a natural-language answer using an LLM."
                )
            else:
                st.warning(
                    "No matching terms were found. Try different wording."
                )

    st.divider()

    st.header("Document Corpus Explorer")

    corpus_df = pd.DataFrame([
        {
            "Document ID": doc["id"],
            "Document title": doc["title"],
            "Category": doc["category"],
            "Text length": len(doc["text"]),
        }
        for doc in DOCUMENTS
    ])

    st.dataframe(
        corpus_df,
        use_container_width=True,
        hide_index=True,
    )

    counts = (
        corpus_df["Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Document count")
    )

    st.subheader("Corpus composition")
    st.bar_chart(counts.set_index("Category"))


# ============================================================
# TAB 2: VAE LABORATORY - CLEARLY MARKED PROTOTYPE
# ============================================================

with vae:
    st.header("VAE Anomaly Laboratory")

    st.warning(
        "Prototype only: this tab does not yet contain a trained "
        "Variational Autoencoder or reconstruction-error calculation."
    )

    st.write(
        "Adjust synthetic telemetry values to explore how unusual "
        "equipment conditions might provide additional RAG context."
    )

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Telemetry controls")

        temperature = st.slider(
            "Temperature (°C)",
            20, 100, 55,
            key="vae_temperature",
        )

        packet_loss = st.slider(
            "Packet loss (%)",
            0, 50, 5,
            key="vae_packet_loss",
        )

        restarts = st.slider(
            "Restart count",
            0, 10, 1,
            key="vae_restarts",
        )

    with right:
        st.subheader("Current measurements")

        m1, m2, m3 = st.columns(3)

        m1.metric("Temperature", f"{temperature} °C")
        m2.metric("Packet loss", f"{packet_loss}%")
        m3.metric("Restarts", restarts)

        # Illustrative rule-based thresholds only.
        flagged = (
            temperature >= 80
            or packet_loss >= 20
            or restarts >= 4
        )

        if flagged:
            st.error(
                "Illustrative threshold exceeded. "
                "This is not a trained anomaly detector."
            )
        else:
            st.success(
                "No illustrative threshold exceeded."
            )

        telemetry_df = pd.DataFrame({
            "Measurement": [
                "Temperature (°C)",
                "Packet loss (%)",
                "Restart count",
            ],
            "Current value": [
                temperature,
                packet_loss,
                restarts,
            ],
        })

        st.bar_chart(
            telemetry_df.set_index("Measurement")
        )

    st.divider()

    st.subheader("Proposed connection to RAG")

    st.write(
        "A future trained VAE could calculate reconstruction error "
        "for telemetry. If validated as useful, its anomaly score could "
        "prioritize maintenance documents relevant to the detected "
        "condition."
    )


# ============================================================
# TAB 3: NAIVE BAYES
# ============================================================

with nb:
    st.header("Naive Bayes Query Intelligence")

    st.write(
        "Classify the intent of a maintenance question and examine "
        "how the predicted category could influence retrieval."
    )

    nb_question = st.text_input(
        "Enter a technical question",
        placeholder="Why does the equipment repeatedly restart?",
        key="nb_question",
    )

    if st.button(
        "CLASSIFY QUERY",
        type="primary",
        use_container_width=True,
        key="nb_run",
    ):
        if not nb_question.strip():
            st.warning("Enter a question first.")
        else:
            predicted, probabilities = classify_query(nb_question)

            st.subheader("Predicted category")

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

            st.subheader("Class probability distribution")

            st.bar_chart(
                prob_df.set_index("Category")
            )

            st.caption(
                "The classifier uses a small synthetic training set. "
                "Its probabilities are not calibrated confidence estimates."
            )

            st.divider()

            st.subheader("Category-aware retrieval")

            baseline_results = retrieve_documents(
                nb_question,
                top_k=len(DOCUMENTS),
            )

            routed_results = retrieve_with_category(
                nb_question,
                predicted,
                top_k=5,
            )

            left, right = st.columns(2)

            with left:
                st.markdown("#### Baseline ranking")
                st.caption("Text similarity only.")

                display_results(
                    baseline_results[:3]
                )

            with right:
                st.markdown("#### Category-aware ranking")
                st.caption(
                    "Text similarity plus a category bonus."
                )

                display_results(
                    routed_results[:3],
                    show_adjusted=True,
                )

            st.info(
                "The fixed category bonus is a demonstration heuristic. "
                "It can improve or worsen ranking and requires evaluation."
            )

            with st.expander("View synthetic training examples"):
                training_df = pd.DataFrame(
                    TRAINING_DATA,
                    columns=["Training question", "Category"],
                )

                st.dataframe(
                    training_df,
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# TAB 4: HMM STATE TRACKER
# ============================================================

with hmm:
    st.header("Hidden Markov Model State Tracker")

    st.write(
        "Use the Viterbi algorithm to infer the most likely hidden "
        "equipment-state sequence from observations."
    )

    st.warning(
        "The HMM transition and emission probabilities are illustrative. "
        "They have not been trained on real equipment data."
    )

    observations = st.multiselect(
        "Select observations in chronological order",
        HMM_OBSERVATIONS,
        default=[
            "Normal temperature",
            "Elevated temperature",
            "High packet loss",
            "Repeated restart",
        ],
        key="hmm_observations",
    )

    if observations:
        states, log_probability = viterbi_decode(observations)

        st.subheader("Inferred state sequence")

        state_df = pd.DataFrame({
            "Time step": range(1, len(observations) + 1),
            "Observation": observations,
            "Inferred state": states,
        })

        st.dataframe(
            state_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("State transition path")

        st.markdown(
            " → ".join(f"**{state}**" for state in states)
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Final inferred state",
            states[-1],
        )

        col2.metric(
            "Best-path log probability",
            f"{log_probability:.3f}",
        )

        st.caption(
            "The log probability scores the most likely path under "
            "the assumed model. It is not a calibrated fault probability."
        )

        st.subheader("Transition probability matrix")

        transition_df = pd.DataFrame(
            HMM_TRANSITION,
            index=HMM_STATES,
            columns=HMM_STATES,
        )

        st.dataframe(
            transition_df.style.format("{:.2f}"),
            use_container_width=True,
        )

        if states[-1] == "Fault":
            retrieval_category = "fault_isolation"
            st.error(
                "Inferred state: Fault. Fault-isolation documents "
                "could be prioritized for further investigation."
            )

        elif states[-1] == "Degraded":
            retrieval_category = "troubleshooting"
            st.warning(
                "Inferred state: Degraded. Troubleshooting documents "
                "could be prioritized."
            )

        else:
            retrieval_category = "preventive_maintenance"
            st.success(
                "Inferred state: Normal. Preventive-maintenance "
                "documents could be prioritized."
            )

        st.divider()

        st.subheader("Retrieve documents using state context")

        if st.button(
            "RETRIEVE STATE-RELEVANT DOCUMENTS",
            type="primary",
            use_container_width=True,
            key="hmm_rag_search",
        ):
            query_text = " ".join(observations)

            results = retrieve_documents(
                query_text,
                top_k=len(DOCUMENTS),
            )

            for result in results:
                bonus = (
                    0.15
                    if result["category"] == retrieval_category
                    else 0.0
                )

                result["category_bonus"] = bonus
                result["adjusted_score"] = result["score"] + bonus

            results.sort(
                key=lambda item: item["adjusted_score"],
                reverse=True,
            )

            display_results(
                results[:3],
                show_adjusted=True,
            )

            st.caption(
                "State-aware retrieval uses a fixed category bonus. "
                "The inferred state and retrieved evidence must be "
                "validated independently."
            )

    else:
        st.info(
            "Select one or more observations to run Viterbi decoding."
        )


# ============================================================
# TAB 5: EVALUATION
# ============================================================

with evaluation:
    st.header("Comparative RAG Evaluation")

    st.write(
        "A framework for testing whether model-assisted retrieval "
        "improves results over the baseline."
    )

    status_df = pd.DataFrame([
        {
            "Component": "TF-IDF retrieval",
            "Status": "Implemented",
            "Purpose": "Baseline document retrieval",
        },
        {
            "Component": "Naive Bayes",
            "Status": "Implemented",
            "Purpose": "Query intent classification",
        },
        {
            "Component": "Category-aware reranking",
            "Status": "Implemented",
            "Purpose": "Heuristic reranking",
        },
        {
            "Component": "HMM / Viterbi",
            "Status": "Implemented",
            "Purpose": "Hidden-state sequence inference",
        },
        {
            "Component": "VAE",
            "Status": "Not implemented",
            "Purpose": "Future anomaly detection",
        },
        {
            "Component": "Generative LLM",
            "Status": "Not connected",
            "Purpose": "Future answer generation",
        },
    ])

    st.subheader("Implementation status")

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Recommended retrieval metrics")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Recall@k", "Planned")
    metric2.metric("MRR", "Planned")
    metric3.metric("Precision@k", "Planned")

    st.markdown("""
    **Recall@k:** Does the top-k set contain the relevant document?

    **Mean Reciprocal Rank (MRR):** How high is the first relevant result?

    **Precision@k:** What proportion of the top-k results are relevant?
    """)

    st.warning(
        "A higher similarity score does not prove better retrieval. "
        "Evaluation requires a fixed set of test questions with "
        "ground-truth relevant document IDs."
    )

    st.subheader("Suggested experiment")

    st.write(
        "Create a held-out set of synthetic questions, label the "
        "relevant document IDs, and compare baseline TF-IDF with "
        "category-aware retrieval on exactly the same queries."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="lab-footer">
        <strong>Intelligent RAG Model Laboratory</strong><br>
        Research prototype · Synthetic data only ·
        Not for operational maintenance decisions
    </div>
    """,
    unsafe_allow_html=True,
)
